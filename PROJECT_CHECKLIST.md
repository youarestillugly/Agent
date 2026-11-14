# DAT404 Project Checklist

## A. Problem and architecture
- [ ] State the problem, target users and operating environment.
- [ ] Explain why an agent is suitable (not merely a chat interface).
- [ ] Complete the PEAS table.
- [ ] Explain the architecture and agent loop.
- [ ] Include and explain the architecture diagram.

## B. Implementation and functionality
- [ ] Upload and parse CSV files.
- [ ] Upload Excel files and select a worksheet.
- [ ] Run Qwen3 through Ollama.
- [ ] Validate LLM-selected tools against the allow-list.
- [ ] Demonstrate at least two meaningful tools (this project includes four).
- [ ] Show multi-step planning and execution trace.
- [ ] Store episodic memory in SQLite.
- [ ] Collect user feedback and show feedback summary/adaptation.
- [ ] Handle invalid files, unavailable Ollama and malformed model output.
- [ ] Verify that original data is not overwritten.

## C. Technologies
- [ ] Explain why Ollama is used for local LLM inference.
- [ ] Explain why Qwen3 is used for planning and explanation.
- [ ] Explain the role of Python, pandas, Streamlit and SQLite.
- [ ] Explain why deterministic tools verify data facts.
- [ ] Discuss privacy and local-service considerations.

## D. Evaluation and visualisation
- [ ] Prepare at least 8 test cases (include normal, complex and edge cases).
- [ ] Record expected and actual tool selections.
- [ ] Calculate tool execution success rate.
- [ ] Calculate task completion rate.
- [ ] Measure response latency.
- [ ] Record user feedback scores.
- [ ] Include charts and a failure analysis.
- [ ] Do not invent results; report measured results only.

## E. Documentation and defence
- [ ] Complete the technical report using `report/TECHNICAL_REPORT_TEMPLATE.md`.
- [ ] Add references and dataset attribution.
- [ ] Add screenshots from your own working application.
- [ ] Keep Git commits showing progressive development.
- [ ] Practise the 10-minute demonstration.
- [ ] Practise the 5-minute defence questions.
- [ ] Be able to explain each function and important code block.

## Final submission package
- [ ] Source code
- [ ] README and setup instructions
- [ ] Technical report
- [ ] Architecture diagram
- [ ] Demonstration-ready app
- [ ] Evaluation test cases and measured results
- [ ] Evaluation visualisations
- [ ] Safety tests and limitations
- [ ] References and attribution
- [ ] Evidence of progressive development
