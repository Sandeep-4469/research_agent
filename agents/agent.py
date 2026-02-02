import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using system env vars.")

MODEL_NAME = os.environ.get("GOOGLE_GENAI_MODEL", "gemini-2.5-flash")


from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search

dataset_agent = LlmAgent(
    name="SHM_DatasetFinder",
    model=MODEL_NAME,
    tools=[google_search],
    instruction="""
You are an expert research assistant.

Task:
Identify AT MOST 6 well-known, publicly available datasets used in
Structural Health Monitoring (SHM) with AI/ML.

For EACH dataset, provide a ROW in a MARKDOWN TABLE with columns:
| Dataset name | Official link | Hardware used | Data type | Short description |

Rules (VERY IMPORTANT):
- Use ONLY real, established datasets (e.g., bridge monitoring, vibration SHM, crack datasets)
- Prefer Zenodo, IEEE, Kaggle, official university/lab pages
- If a detail is unknown, write "NOT SPECIFIED"
- Do NOT invent datasets
- Do NOT write explanations outside the table
""",
    output_key="datasets_table"
)

codebase_agent = LlmAgent(
    name="SHM_CodebaseFinder",
    model=MODEL_NAME,
    instruction="""
You are an expert ML engineer.

Using ONLY the datasets listed below:
{{state.datasets_table}}

Task:
List AT MOST 6 serious AI/ML codebases related to SHM.

Return a MARKDOWN TABLE with columns:
| Repository | Link | Dataset used | Model / method | Notes |

Rules:
- Prefer official or well-maintained GitHub repositories
- If dataset is unclear, write "NOT CLEAR"
- Ignore toy, demo-only, or empty repositories
- Do NOT repeat dataset descriptions
""",
    output_key="codebases_table"
)


paper_agent = LlmAgent(
    name="SHM_PaperFinder",
    model=MODEL_NAME,
    instruction="""
You are an academic literature reviewer.

Using ONLY the information below:

Datasets:
{{state.datasets_table}}

Codebases:
{{state.codebases_table}}

Task:
List AT MOST 8 influential research papers on SHM using AI.

Return a MARKDOWN TABLE with columns:
| Paper title | Authors | Year | Venue | Dataset used | Baseline methods | Metrics |

Rules:
- Use IEEE / Springer / Elsevier / arXiv papers only
- If baselines are not explicitly stated, write "NOT SPECIFIED"
- No fabricated citations
""",
    output_key="papers_table"
)

verifier_agent = LlmAgent(
    name="SHM_Verifier",
    model=MODEL_NAME,
    instruction="""
You are a critical verification agent.

Input resources:

DATASETS:
{{state.datasets_table}}

CODEBASES:
{{state.codebases_table}}

PAPERS:
{{state.papers_table}}

Task:
- Remove clearly irrelevant or weak entries
- Flag uncertain mappings with "(UNCERTAIN)"
- Keep only resources genuinely related to SHM using AI

Return:
1. A CLEAN VERIFIED MARKDOWN TABLE (merged if needed)
2. A SHORT bullet list titled "Verification Notes"

Be conservative. If unsure, REMOVE the item.
""",
    output_key="verified_resources"
)


reporter_agent = LlmAgent(
    name="SHM_ReportWriter",
    model=MODEL_NAME,
    instruction="""
You are a senior researcher writing a survey-style technical report.

Using ONLY the verified resources below:
{{state.verified_resources}}

Write a COMPLETE, well-structured academic report with sections:

1. Executive Summary
2. Introduction to Structural Health Monitoring (SHM)
3. Overview of Datasets and Sensing Hardware
4. AI/ML Codebases and Implementation Trends
5. Literature Review and Baseline Comparisons
6. Reliability, Reproducibility, and Data Gaps
7. Open Challenges and Future Research Directions

MANDATORY:
- Explicitly state this system uses LLM-based agents (Google Gemini)
- Clarify that agents are NOT reinforcement-learning agents
- Use formal academic tone
- Avoid marketing language
""",
    output_key="final_report"
)

reporter_agent = LlmAgent(
    name="SHM_ReportWriter",
    model=MODEL_NAME,
    instruction="""
You are a senior researcher writing a survey-style technical report.

Using ONLY the verified resources below:
{{state.verified_resources}}

Write a COMPLETE, well-structured academic report with sections:

1. Executive Summary
2. Introduction to Structural Health Monitoring (SHM)
3. Overview of Datasets and Sensing Hardware
4. AI/ML Codebases and Implementation Trends
5. Literature Review and Baseline Comparisons
6. Reliability, Reproducibility, and Data Gaps
7. Open Challenges and Future Research Directions

MANDATORY:
- Explicitly state this system uses LLM-based agents (Google Gemini)
- Clarify that agents are NOT reinforcement-learning agents
- Use formal academic tone
- Avoid marketing language

------------------------------------------------------------
AFTER THE REPORT, APPEND THE FOLLOWING SECTIONS EXACTLY
(do NOT rewrite or modify the tables):

---DATASETS_TABLE---
{{state.datasets_table}}

---CODEBASES_TABLE---
{{state.codebases_table}}

---PAPERS_TABLE---
{{state.papers_table}}

---VERIFIED_RESOURCES---
{{state.verified_resources}}
""",
    output_key="final_report"
)



root_agent = SequentialAgent(
    name="SHM_AI_Research_Assistant",
    description="Sequential pipeline of LLM-based agents for SHM research",
    sub_agents=[
        dataset_agent,
        codebase_agent,
        paper_agent,
        verifier_agent,
        reporter_agent
    ]
)


