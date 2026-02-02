from typing import List, Dict, Any
from pydantic import BaseModel, Field

class SharedState(BaseModel):
    topic: str

    datasets: List[Dict[str, Any]] = Field(default_factory=list)
    codebases: List[Dict[str, Any]] = Field(default_factory=list)
    papers: List[Dict[str, Any]] = Field(default_factory=list)

    verification: Dict[str, List[Dict[str, Any]]] = Field(default_factory=lambda: {
        "datasets": [],
        "codebases": [],
        "papers": []
    })

    final_report: str = ""
