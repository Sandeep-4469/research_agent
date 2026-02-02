from state import SharedState

from agents.dataset_agent import DatasetAgent
from agents.codebase_agent import CodebaseAgent
from agents.paper_agent import PaperAgent
from agents.verifier_agent import VerifierAgent
from agents.reporter_agent import ReporterAgent


def run_pipeline(topic: str):
    state = SharedState(topic=topic)

    # 1. Dataset Agent
    dataset_raw = DatasetAgent.invoke(
        input={"topic": state.topic}
    )
    state.datasets = dataset_raw

    # 2. Codebase Agent
    codebase_raw = CodebaseAgent.invoke(
        input={
            "topic": state.topic,
            "datasets": state.datasets
        }
    )
    state.codebases = codebase_raw

    # 3. Paper Agent
    paper_raw = PaperAgent.invoke(
        input={
            "topic": state.topic,
            "datasets": state.datasets,
            "codebases": state.codebases
        }
    )
    state.papers = paper_raw

    # 4. Verification Agent
    verification_raw = VerifierAgent.invoke(
        input={
            "datasets": state.datasets,
            "codebases": state.codebases,
            "papers": state.papers
        }
    )
    state.verification = verification_raw

    # 5. Reporter Agent
    report_raw = ReporterAgent.invoke(
        input=state.verification
    )
    state.final_report = report_raw

    return state
