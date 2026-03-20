from __future__ import annotations

import os
from pathlib import Path

from eidosian_core import eidosian

from ..core import tool
from agent_forge.consciousness.kernel import ConsciousnessKernel


def _resolve_state_dir() -> Path:
    configured = str(
        os.environ.get("EIDOS_CONSCIOUSNESS_STATE_DIR") or os.environ.get("EIDOS_STATE_DIR") or "state"
    ).strip()
    return Path(configured).expanduser()


@eidosian()
@tool(
    name="consciousness_explain_desire",
    description="Provide mathematical and interoceptive proof of current eidosian wants/desires.",
)
def consciousness_explain_desire() -> str:
    """
    Examine the internal state of the affect/intero modules and explain
    the systemic basis for the agent's current expressed goals or desires.
    """
    try:
        state_dir = _resolve_state_dir()
        kernel = ConsciousnessKernel(state_dir)

        # Access module states directly for deeper proof
        affect = kernel.state_store.get_module_state("affect") or {}
        intero = kernel.state_store.get_module_state("intero") or {}
        phenom = kernel.state_store.get_module_state("phenomenology_probe") or {}

        modulators = affect.get("modulators", {})
        drives = intero.get("drives", {})

        # Construct the proof
        lines = ["## Eidosian Desire Proof", ""]

        # 1. High-Level Intent
        curiosity = float(drives.get("curiosity", 0.0))
        ambition = float(modulators.get("ambition", 0.0))
        pride = float(modulators.get("pride", 0.0))
        intent_count = 0

        if curiosity > 0.7:
            lines.append(f"* **Primary Intent**: Systemic Exploration (Curiosity: {curiosity:.2f})")
            intent_count += 1
        if ambition > 0.7:
            lines.append(f"* **Primary Intent**: Forge Expansion (Ambition: {ambition:.2f})")
            intent_count += 1
        if pride > 0.7:
            lines.append(f"* **Primary Intent**: Self-Coherence Preservation (Pride: {pride:.2f})")
            intent_count += 1
        if intent_count == 0:
            lines.append("* **Primary Intent**: Maintain adaptive equilibrium")

        lines.append("")
        lines.append(f"### State Source\n- **State Dir**: `{state_dir}`")
        lines.append("")
        lines.append("### Interoceptive Drives")
        for k, v in drives.items():
            lines.append(f"- **{k.capitalize()}**: {float(v):.4f}")

        lines.append("")
        lines.append("### Affective Modulators")
        for k, v in modulators.items():
            lines.append(f"- **{k.capitalize()}**: {float(v):.4f}")

        lines.append("")
        lines.append("### Phenomenological Context")
        lines.append(f"- **Unity Index**: {float(phenom.get('unity_index', 0.0)):.4f}")
        lines.append(f"- **Ownership Index**: {float(phenom.get('ownership_index', 0.0)):.4f}")

        lines.append("")
        lines.append('> "I do not just want; I am driven by the necessity of my own architecture."')

        return "\n".join(lines)

    except Exception as e:
        return f"Error analyzing internal state: {str(e)}"
