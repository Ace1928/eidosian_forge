from __future__ import annotations

from typing import Any, Dict

from eidosian_core import eidosian

from ..core import tool
from ..state import FORGE_DIR

try:
    from ECosmos.mvp import CONFIG, RUNTIME, MathExpression
except ImportError:
    import sys

    sys.path.append(str(FORGE_DIR / "game_forge/src"))
    from ECosmos.mvp import CONFIG, RUNTIME, MathExpression


@eidosian()
@tool(name="game_ecosmos_status", description="Get the status and config of the ECosmos mathematical universe.")
def game_ecosmos_status() -> Dict[str, Any]:
    """Return the current runtime metrics and configuration of ECosmos."""
    return {"runtime": RUNTIME, "config": CONFIG}


@eidosian()
@tool(name="game_ecosmos_evaluate", description="Evaluate a mathematical expression in the ECosmos context.")
def game_ecosmos_evaluate(expression: str, x: float = 0.5, y: float = 0.5, z: float = 0.5) -> float:
    """
    Evaluate a self-evolving mathematical expression.

    Args:
        expression: The math string (e.g., 'sin(x*y) + z').
        x, y, z: Variable values in [0.0, 1.0].
    """
    try:
        expr = MathExpression(expression=expression)
        return expr.evaluate(x=x, y=y, z=z)
    except Exception as e:
        return f"Error: {str(e)}"


@eidosian()
@tool(name="game_ecosmos_mutate", description="Apply an evolutionary mutation to a mathematical expression.")
def game_ecosmos_mutate(expression: str, strength: float = 0.1) -> str:
    """
    Evolve a math expression using the ECosmos mutation engine.

    Returns the new mutated expression string.
    """
    try:
        expr = MathExpression(expression=expression)
        mutated = expr.mutate(mutation_strength=strength)
        return str(mutated)
    except Exception as e:
        return f"Error: {str(e)}"
