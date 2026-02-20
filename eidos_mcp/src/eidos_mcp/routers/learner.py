from __future__ import annotations

import json
from pathlib import Path

from eidosian_core import eidosian

from ..core import tool
from ..state import FORGE_DIR

_LEARNER_IMPORT_ERROR = None
EidosianLearner = None
for _import_path in (
    "managed_agents.learner.core",
    "agent_forge.managed_agents.learner.core",
):
    try:
        module = __import__(_import_path, fromlist=["EidosianLearner"])
        EidosianLearner = getattr(module, "EidosianLearner")
        break
    except Exception as exc:  # pragma: no cover - optional integration
        _LEARNER_IMPORT_ERROR = exc

learner = None
if EidosianLearner is not None:
    config_path = FORGE_DIR / "agent_forge" / "managed_agents" / "learner" / "config.yaml"
    try:
        learner = EidosianLearner(Path(config_path))
    except Exception as exc:  # pragma: no cover - defensive
        _LEARNER_IMPORT_ERROR = exc


def _learner_unavailable_response() -> str:
    return json.dumps(
        {
            "ok": False,
            "error": "learner_unavailable",
            "detail": str(_LEARNER_IMPORT_ERROR) if _LEARNER_IMPORT_ERROR else "unknown",
        }
    )


@tool(
    name="learner_run_mission",
    description="Task the Eidosian Learner with a recursive self-improvement mission.",
    parameters={
        "type": "object",
        "properties": {
            "objective": {"type": "string", "description": "The learning goal."},
            "max_steps": {"type": "integer", "default": 5},
        },
        "required": ["objective"],
    },
)
@eidosian()
async def learner_run_mission(objective: str, max_steps: int = 5) -> str:
    """Run a learning mission."""
    if learner is None:
        return _learner_unavailable_response()
    return await learner.run_mission(objective, max_steps)


@tool(
    name="learner_reflect_docs",
    description="Ask the Eidosian Learner to critique a documentation file.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Relative path to the .md file."}},
        "required": ["path"],
    },
)
@eidosian()
async def learner_reflect_docs(path: str) -> str:
    """Critique docs."""
    if learner is None:
        return _learner_unavailable_response()
    # Re-using mission logic for reflection if needed, or keeping specialized method
    # For now, we wrap it in a mission for better tracking
    return await learner.run_mission(f"Critique documentation at {path}", max_steps=3)
