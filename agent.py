
"""
AI-Powered Data Quality Detective
LangGraph-based intelligent agent.

Framework: LangGraph
LLM: Qwen3 through Ollama
Memory: Existing SQLite AgentMemory
Tools: Existing tools.py
"""

from __future__ import annotations

import json
import re
import time

from typing import Any, TypedDict

import pandas as pd
import requests

from langgraph.graph import StateGraph, START, END

from tools import (
    profile_dataset,
    detect_quality_issues,
    detect_anomalies,
    create_cleaning_preview,
)

from memory import AgentMemory


# =========================================================
# 1. AVAILABLE PYTHON TOOLS
# =========================================================

AVAILABLE_TOOLS = {
    "profile_dataset": profile_dataset,
    "detect_quality_issues": detect_quality_issues,
    "detect_anomalies": detect_anomalies,
    "create_cleaning_preview": create_cleaning_preview,
}


# =========================================================
# 2. LANGGRAPH SHARED STATE
# =========================================================

class InvestigationState(TypedDict, total=False):

    # Input
    df: Any
    user_goal: str

    # Planning
    plan: dict
    steps: list

    # Tool execution
    observations: dict
    trace: list

    # Explanation
    summary: str

    # Timing and status
    started_at: float
    latency_seconds: float
    error: str

    # Final output
    final_result: dict


# =========================================================
# 3. DATA QUALITY AGENT
# =========================================================

class DataQualityAgent:

    def __init__(
        self,
        model: str = "qwen3:8b",
        ollama_url: str = "http://localhost:11434",
        memory_path: str = "outputs/agent_memory.db",
    ):

        self.model = model
        self.ollama_url = ollama_url.rstrip("/")

        self.memory = AgentMemory(memory_path)

        self.trace = []

        # Build the actual LangGraph workflow.
        self.graph = self._build_graph()

    # =====================================================
    # 4. OLLAMA CONNECTION
    # =====================================================

    def _ask_qwen(
        self,
        prompt: str,
        temperature: float = 0.1,
    ) -> str:

        """Send a prompt to the local Qwen3 model."""

        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 512,
                    "num_ctx": 4096,
                },
            },
            timeout=300,
        )

        response.raise_for_status()

        content = response.json()["message"]["content"]

        content = re.sub(
            r"<think>.*?</think>",
            "",
            content,
            flags=re.S,
        ).strip()

        return content

    # =====================================================
    # 5. PLANNING LOGIC
    # =====================================================

    def _plan(
        self,
        user_goal: str,
        columns: list[str],
    ) -> dict:

        """Ask Qwen3 to select tools for the investigation."""

        feedback = self.feedback_guidance()

        prompt = f"""
You are the planning component of an intelligent
Data Quality Detective agent.

Your task is to create a safe investigation plan.

AVAILABLE TOOLS:

1. profile_dataset
   Understand dataset structure, dimensions and statistics.

2. detect_quality_issues
   Find missing values, duplicate rows, constant columns
   and blank text values.

3. detect_anomalies
   Identify numerical outlier candidates using IQR.

4. create_cleaning_preview
   Suggest non-destructive data cleaning actions.

IMPORTANT RULES:

- Only use the four exact tool names.
- Do not invent tools or columns.
- Select tools relevant to the user's request.
- For a general investigation, include all four tools.
- Do not modify the uploaded dataset.
- Outliers are candidates, not confirmed errors.

USER REQUEST:

{user_goal}

DATASET COLUMNS:

{columns}

PREVIOUS USER FEEDBACK:

{feedback}

Return ONLY valid JSON in this format:

{{
    "reasoning": "Short explanation of the plan",
    "steps": [
        {{
            "tool": "profile_dataset",
            "purpose": "Understand the dataset"
        }},
        {{
            "tool": "detect_quality_issues",
            "purpose": "Check data quality"
        }}
    ]
}}
"""

        try:

            raw = self._ask_qwen(prompt)

            raw = re.sub(
                r"^```(?:json)?|```$",
                "",
                raw,
                flags=re.M,
            ).strip()

            match = re.search(
                r"\{.*\}",
                raw,
                flags=re.S,
            )

            if not match:
                raise ValueError("No JSON plan returned by Qwen3")

            plan = json.loads(match.group())

            safe_steps = []
            already_selected = set()

            for step in plan.get("steps", []):

                if not isinstance(step, dict):
                    continue

                name = step.get("tool")

                if (
                    name in AVAILABLE_TOOLS
                    and name not in already_selected
                ):

                    safe_steps.append({
                        "tool": name,
                        "purpose": str(
                            step.get("purpose", "")
                        )[:300],
                    })

                    already_selected.add(name)

            if not safe_steps:
                raise ValueError("No valid tools in plan")

            return {
                "reasoning": str(
                    plan.get("reasoning", "")
                ),
                "steps": safe_steps,
                "framework": "LangGraph",
                "fallback": False,
            }

        except Exception as exc:

            # Safe fallback if Qwen3 is unavailable
            # or returns an invalid plan.

            return {
                "reasoning": (
                    "A predefined safe investigation plan "
                    "was selected because the LLM plan "
                    f"was unavailable or invalid: {exc}"
                ),
                "steps": [
                    {
                        "tool": "profile_dataset",
                        "purpose": "Profile the dataset",
                    },
                    {
                        "tool": "detect_quality_issues",
                        "purpose": "Identify quality problems",
                    },
                    {
                        "tool": "detect_anomalies",
                        "purpose": "Identify potential outliers",
                    },
                    {
                        "tool": "create_cleaning_preview",
                        "purpose": "Suggest cleaning actions",
                    },
                ],
                "framework": "LangGraph",
                "fallback": True,
            }

    # =====================================================
    # 6. NODE 1: PLANNING
    # =====================================================

    def planning_node(
        self,
        state: InvestigationState,
    ) -> dict:

        """LangGraph planning node."""

        df = state["df"]

        user_goal = state["user_goal"]

        columns = [
            str(column)
            for column in df.columns
        ]

        plan = self._plan(
            user_goal,
            columns,
        )

        trace = state.get("trace", []).copy()

        trace.append({
            "node": "planning_node",
            "framework": "LangGraph",
            "status": "success",
            "description": (
                "Qwen3 generated and validated the "
                "investigation plan."
            ),
        })

        return {
            "plan": plan,
            "steps": plan["steps"],
            "trace": trace,
        }

    # =====================================================
    # 7. NODE 2: TOOL EXECUTION
    # =====================================================

    def tool_execution_node(
        self,
        state: InvestigationState,
    ) -> dict:

        """LangGraph node for executing Python tools."""

        df = state["df"]

        observations = {}

        trace = state.get("trace", []).copy()

        for step in state.get("steps", []):

            tool_name = step["tool"]

            tool = AVAILABLE_TOOLS[tool_name]

            try:

                result = tool(df)

                observations[tool_name] = result

                trace.append({
                    "node": "tool_execution_node",
                    "tool": tool_name,
                    "status": "success",
                    "purpose": step["purpose"],
                })

            except Exception as exc:

                observations[tool_name] = {
                    "error": str(exc)
                }

                trace.append({
                    "node": "tool_execution_node",
                    "tool": tool_name,
                    "status": "failed",
                    "error": str(exc),
                })

        trace.append({
            "node": "tool_execution_node",
            "status": "success",
            "description": (
                "LangGraph executed the selected "
                "Python data analysis tools."
            ),
        })

        return {
            "observations": observations,
            "trace": trace,
        }

    # =====================================================
    # 8. NODE 3: EXPLANATION
    # =====================================================

    def explanation_node(
        self,
        state: InvestigationState,
    ) -> dict:

        """LangGraph explanation node."""

        observations = state.get(
            "observations", {}
        )

        prompt = f"""
You are the explanation component of an
AI-Powered Data Quality Detective.

Explain the dataset investigation findings clearly.

USER REQUEST:

{state.get("user_goal", "")}

VERIFIED PYTHON TOOL RESULTS:

{json.dumps(observations, default=str)[:14000]}

INSTRUCTIONS:

1. Explain the important findings.
2. Identify missing values and duplicates.
3. Discuss numerical outlier candidates.
4. Give practical recommendations.
5. Do not invent statistics.
6. Do not claim an outlier is definitely an error.
7. Explain that cleaning should be reviewed by the user.
8. Use simple, professional language.
"""

        try:

            summary = self._ask_qwen(prompt)

        except Exception as exc:

            summary = (
                "The Python analysis tools completed, but "
                "Qwen3 could not generate an explanation. "
                f"Reason: {exc}. "
                "Please review the structured observations."
            )

        trace = state.get("trace", []).copy()

        trace.append({
            "node": "explanation_node",
            "status": "success",
            "description": (
                "Qwen3 generated an explanation "
                "from the verified tool observations."
            ),
        })

        return {
            "summary": summary,
            "trace": trace,
        }

    # =====================================================
    # 9. NODE 4: SQLITE MEMORY
    # =====================================================

    def memory_node(
        self,
        state: InvestigationState,
    ) -> dict:

        """LangGraph memory node."""

        elapsed = round(
            time.time() - state["started_at"],
            3,
        )

        trace = state.get("trace", []).copy()

        trace.append({
            "node": "memory_node",
            "status": "success",
            "description": (
                "Investigation episode saved to SQLite memory."
            ),
        })

        result = {
            "goal": state["user_goal"],
            "plan": state.get("plan", {}),
            "observations": state.get(
                "observations", {}
            ),
            "summary": state.get("summary", ""),
            "trace": trace,
            "latency_seconds": elapsed,
            "framework": "LangGraph",
        }

        try:

            self.memory.save_episode(
                state["user_goal"],
                result,
            )

        except Exception as exc:

            trace.append({
                "node": "memory_node",
                "status": "failed",
                "error": str(exc),
            })

            result["trace"] = trace

        return {
            "latency_seconds": elapsed,
            "final_result": result,
            "trace": trace,
        }

    # =====================================================
    # 10. BUILD LANGGRAPH WORKFLOW
    # =====================================================

    def _build_graph(self):

        """Construct and compile the LangGraph StateGraph."""

        workflow = StateGraph(
            InvestigationState
        )

        # Register graph nodes.

        workflow.add_node(
            "planning",
            self.planning_node,
        )

        workflow.add_node(
            "tool_execution",
            self.tool_execution_node,
        )

        workflow.add_node(
            "explanation",
            self.explanation_node,
        )

        workflow.add_node(
            "memory",
            self.memory_node,
        )

        # Define graph connections.

        workflow.add_edge(
            START,
            "planning",
        )

        workflow.add_edge(
            "planning",
            "tool_execution",
        )

        workflow.add_edge(
            "tool_execution",
            "explanation",
        )

        workflow.add_edge(
            "explanation",
            "memory",
        )

        workflow.add_edge(
            "memory",
            END,
        )

        # Compile into an executable graph.

        return workflow.compile()

    # =====================================================
    # 11. RUN INVESTIGATION
    # =====================================================

    def investigate(
        self,
        df: pd.DataFrame,
        user_goal: str = (
            "Investigate this dataset and identify "
            "data quality problems."
        ),
    ) -> dict:

        """Execute the complete LangGraph workflow."""

        self.trace = []

        initial_state = {

            "df": df.copy(deep=True),

            "user_goal": user_goal,

            "observations": {},

            "trace": [],

            "started_at": time.time(),

        }

        # Actual LangGraph execution.

        graph_result = self.graph.invoke(
            initial_state
        )

        result = graph_result["final_result"]

        self.trace = result["trace"]

        return result

    # =====================================================
    # 12. FOLLOW-UP QUESTIONS
    # =====================================================

    def answer_question(
        self,
        question: str,
        latest_result: dict,
    ) -> str:

        """Answer questions using investigation evidence."""

        recent = self.memory.recent_episodes(
            limit=3
        )

        prompt = f"""
You are a data quality assistant.

Answer the user's question using the
verified investigation evidence.

QUESTION:

{question}

LATEST INVESTIGATION:

{json.dumps(latest_result.get('observations', {}), default=str)[:12000]}

RECENT MEMORY:

{json.dumps(recent, default=str)[:3000]}

Do not invent statistics.
If the information is unavailable, say so.
Use clear and simple language.
"""

        try:

            return self._ask_qwen(prompt)

        except Exception as exc:

            return (
                "Qwen3 is currently unavailable. "
                f"Error: {exc}"
            )

    # =====================================================
    # 13. USER FEEDBACK
    # =====================================================

    def record_feedback(
        self,
        rating: int,
        comment: str,
        issue_category: str = "general",
    ) -> None:

        """Store feedback in SQLite memory."""

        self.memory.save_feedback(
            rating,
            comment,
            issue_category,
        )

    # =====================================================
    # 14. MEMORY GUIDANCE
    # =====================================================

    def feedback_guidance(self) -> str:

        """Retrieve feedback guidance from SQLite."""

        try:

            feedback = self.memory.feedback_summary()

            if not feedback:

                return (
                    "No previous feedback available. "
                    "Use balanced investigation priorities."
                )

            return (
                f"Previous feedback: {feedback}. "
                "Consider these concerns when planning "
                "the next investigation."
            )

        except Exception:

            return (
                "Feedback guidance is unavailable. "
                "Use balanced investigation priorities."
            )

    # =====================================================
    # 15. GRAPH VISUALIZATION
    # =====================================================

    def get_graph_mermaid(self) -> str:

        """Return a Mermaid representation of the workflow."""

        return """
graph TD
    A([START]) --> B[Planning Node - Qwen3]
    B --> C[Tool Execution Node]
    C --> D[Explanation Node - Qwen3]
    D --> E[SQLite Memory Node]
    E --> F([END])
        """.strip()
