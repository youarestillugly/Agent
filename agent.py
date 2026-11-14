
"""Ollama-powered Data Quality Detective agent.

The LLM proposes a plan; Python tools perform verifiable operations.
The agent keeps episodic memory and uses feedback to adjust future priorities.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import pandas as pd
import requests

from tools import (
    profile_dataset,
    detect_quality_issues,
    detect_anomalies,
    create_cleaning_preview,
)

from memory import AgentMemory


# ---------------------------------------------------------
# 1. AVAILABLE TOOLS
# ---------------------------------------------------------

AVAILABLE_TOOLS = {
    "profile_dataset": profile_dataset,
    "detect_quality_issues": detect_quality_issues,
    "detect_anomalies": detect_anomalies,
    "create_cleaning_preview": create_cleaning_preview,
}


# ---------------------------------------------------------
# 2. DATA QUALITY AGENT
# ---------------------------------------------------------

class DataQualityAgent:

    def __init__(
        self,
        model: str = "qwen3:8b",
        ollama_url: str = "http://127.0.0.1:11434",
        memory_path: str = "outputs/agent_memory.db",
    ):

        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.memory = AgentMemory(memory_path)
        self.trace = []

        self.session = requests.Session()

    # -----------------------------------------------------
    # 3. COMMUNICATE WITH QWEN3
    # -----------------------------------------------------

    def _ask_qwen(
        self,
        prompt: str,
        temperature: float = 0.1
    ) -> str:

        """Call the locally hosted Qwen3 model through Ollama."""

        try:

            response = self.session.post(
                f"{self.ollama_url}/api/chat",

                json={
                    "model": self.model,

                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    "stream": False,

                    "think": False,

                    "options": {
                        "temperature": temperature,
                        "num_predict": 256,
                        "num_ctx": 2048
                    }
                },

                timeout=300,
            )

            response.raise_for_status()

            data = response.json()

            content = data.get("message", {}).get("content", "")

            if not content.strip():
                raise ValueError("Qwen3 returned an empty response.")

            return content.strip()

        except requests.exceptions.Timeout as exc:

            raise RuntimeError(
                "Qwen3 took too long to respond. "
                "Try a smaller model or reduce the prompt size."
            ) from exc

        except requests.exceptions.ConnectionError as exc:

            raise RuntimeError(
                "Cannot connect to Ollama. "
                "Please make sure the Ollama application is running."
            ) from exc

        except requests.exceptions.HTTPError as exc:

            raise RuntimeError(
                f"Ollama API error: {exc}. "
                "Check that the model is available and Ollama is updated."
            ) from exc

    # -----------------------------------------------------
    # 4. CLEAN AND EXTRACT JSON
    # -----------------------------------------------------

    def _extract_json(self, raw: str) -> dict:

        """Extract a JSON object from the model response."""

        raw = re.sub(
            r"<think>.*?</think>",
            "",
            raw,
            flags=re.S
        ).strip()

        raw = re.sub(
            r"```(?:json)?",
            "",
            raw,
            flags=re.I
        )

        raw = raw.replace("```", "").strip()

        # Find the JSON object in the response.
        start = raw.find("{")
        end = raw.rfind("}")

        if start == -1 or end == -1 or end < start:
            raise ValueError("No valid JSON object found.")

        return json.loads(raw[start:end + 1])

    # -----------------------------------------------------
    # 5. AGENT PLANNING
    # -----------------------------------------------------

    def _plan(
        self,
        user_goal: str,
        columns: list[str]
    ) -> dict[str, Any]:

        """Ask Qwen3 to select tools from a strict allow-list."""

        prompt = f"""
You are the planning component of a data quality agent.

Choose a safe multi-step plan for the user's dataset task.

Available tools and purposes:

- profile_dataset: dataset dimensions, data types and descriptive statistics.
- detect_quality_issues: missing values, duplicate rows, constant columns and suspicious values.
- detect_anomalies: numeric outlier candidates using IQR.
- create_cleaning_preview: propose conservative cleaning actions without changing original data.

Return ONLY valid JSON with these keys:

{{
    "reasoning": "Short explanation of the plan",
    "steps": [
        {{
            "tool": "profile_dataset",
            "purpose": "Understand the dataset"
        }}
    ]
}}

Rules:
1. Only use the four exact tool names listed above.
2. Include profile_dataset for a general investigation.
3. Include detect_quality_issues for a general investigation.
4. Select other relevant tools when necessary.
5. Do not invent dataset columns.
6. Do not include Markdown or additional explanations outside the JSON.

User goal:
{user_goal}

Dataset columns:
{columns}
"""

        try:

            raw = self._ask_qwen(prompt)

            plan = self._extract_json(raw)

            steps = plan.get("steps", [])

            safe_steps = []

            for step in steps:

                if not isinstance(step, dict):
                    continue

                name = step.get("tool")

                if (
                    name in AVAILABLE_TOOLS
                    and name not in [s["tool"] for s in safe_steps]
                ):

                    safe_steps.append(
                        {
                            "tool": name,
                            "purpose": str(
                                step.get("purpose", "")
                            )[:300]
                        }
                    )

            if not safe_steps:
                raise ValueError("No valid tools selected.")

            return {
                "reasoning": str(
                    plan.get("reasoning", "Qwen3 generated a tool plan.")
                ),

                "steps": safe_steps,

                "fallback": False,
            }

        except Exception as exc:

            # Safe deterministic fallback.
            # The agent can still operate without a valid LLM plan.

            return {
                "reasoning": (
                    "Safe fallback plan used because Qwen3 "
                    f"could not generate a valid plan: {str(exc)}"
                ),

                "steps": [
                    {
                        "tool": "profile_dataset",
                        "purpose": "Understand dataset structure"
                    },
                    {
                        "tool": "detect_quality_issues",
                        "purpose": "Inspect common data quality problems"
                    },
                    {
                        "tool": "detect_anomalies",
                        "purpose": "Check numeric columns for outlier candidates"
                    },
                    {
                        "tool": "create_cleaning_preview",
                        "purpose": "Prepare a non-destructive cleaning preview"
                    },
                ],

                "fallback": True,
            }

    # -----------------------------------------------------
    # 6. EXECUTE SELECTED TOOLS
    # -----------------------------------------------------

    def investigate(
        self,
        df: pd.DataFrame,
        user_goal: str = (
            "Investigate this dataset and report its data quality."
        )
    ) -> dict:

        """Perceive -> plan -> select tools -> act -> observe -> reflect."""

        started = time.time()

        self.trace = []

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if df.empty:
            raise ValueError("The uploaded dataset is empty.")

        # STEP 1: Planning
        plan = self._plan(
            user_goal,
            [str(c) for c in df.columns]
        )

        if plan.get("fallback"):
            self.trace.append(
                {
                    "tool": "planning",
                    "status": "fallback",
                    "purpose": plan["reasoning"]
                }
            )
        else:
            self.trace.append(
                {
                    "tool": "planning",
                    "status": "success",
                    "purpose": plan["reasoning"]
                }
            )

        # STEP 2: Tool execution
        observations = {}

        for step in plan["steps"]:

            tool_name = step["tool"]

            tool = AVAILABLE_TOOLS[tool_name]

            try:

                result = tool(df)

                observations[tool_name] = result

                self.trace.append(
                    {
                        "tool": tool_name,
                        "status": "success",
                        "purpose": step["purpose"]
                    }
                )

            except Exception as exc:

                observations[tool_name] = {
                    "error": str(exc)
                }

                self.trace.append(
                    {
                        "tool": tool_name,
                        "status": "failed",
                        "error": str(exc)
                    }
                )

        # STEP 3: Generate explanation
        summary = self._synthesise(
            user_goal,
            observations
        )

        # STEP 4: Record execution time
        elapsed = round(
            time.time() - started,
            3
        )

        # STEP 5: Final result
        result = {

            "goal": user_goal,

            "plan": plan,

            "observations": observations,

            "summary": summary,

            "trace": self.trace,

            "latency_seconds": elapsed,
        }

        # STEP 6: Save episodic memory
        self.memory.save_episode(
            user_goal,
            result
        )

        return result

    # -----------------------------------------------------
    # 7. SYNTHESISE FINDINGS
    # -----------------------------------------------------

    def _synthesise(
        self,
        goal: str,
        observations: dict
    ) -> str:

        """Use Qwen3 to explain verified Python tool results."""

        prompt = f"""
You are Qwen3, the explanation component of a data quality agent.

Explain the findings in clear, simple and cautious language.

Instructions:

1. Explain the important data quality findings.
2. Distinguish confirmed facts from possible anomalies.
3. Never claim that an outlier is definitely an error.
4. Do not invent statistics or dataset information.
5. Recommend practical next steps.
6. Preserve the original dataset.
7. Use headings and bullet points where helpful.
8. Keep the explanation concise.

User goal:
{goal}

Tool observations (the only source of dataset facts):

{json.dumps(observations, default=str)[:10000]}
"""

        try:

            return self._ask_qwen(prompt)

        except Exception as exc:

            return (
                "Qwen3 explanation unavailable. "
                "The Python tool results are still available. "
                f"Reason: {str(exc)}"
            )

    # -----------------------------------------------------
    # 8. FOLLOW-UP QUESTIONS
    # -----------------------------------------------------

    def answer_question(
        self,
        question: str,
        latest_result: dict
    ) -> str:

        """Answer follow-up questions using tool evidence and memory."""

        recent = self.memory.recent_episodes(limit=3)

        prompt = f"""
Answer the user's question using only the evidence below.

If the evidence is insufficient, say so and suggest
which investigation would help.

Do not fabricate values.

Question:
{question}

Latest verified tool observations:

{json.dumps(latest_result.get('observations', {}), default=str)[:8000]}

Recent episodic memory:

{json.dumps(recent, default=str)[:2000]}
"""

        try:

            return self._ask_qwen(prompt)

        except Exception as exc:

            return (
                f"Qwen3 is unavailable: {str(exc)}"
            )

    # -----------------------------------------------------
    # 9. RECORD USER FEEDBACK
    # -----------------------------------------------------

    def record_feedback(
        self,
        rating: int,
        comment: str,
        issue_category: str = "general"
    ) -> None:

        """Save user feedback for future adaptation."""

        self.memory.save_feedback(
            rating,
            comment,
            issue_category
        )

    # -----------------------------------------------------
    # 10. ADAPT FUTURE PRIORITIES
    # -----------------------------------------------------

    def feedback_guidance(self) -> str:

        """Prioritise categories users previously found useful."""

        feedback = self.memory.feedback_summary()

        if not feedback:

            return (
                "No feedback history yet; "
                "use balanced investigation priorities."
            )

        return (
            "Past user feedback summary: "
            f"{feedback}. "
            "Prioritise these concerns in the next explanation."
        )
