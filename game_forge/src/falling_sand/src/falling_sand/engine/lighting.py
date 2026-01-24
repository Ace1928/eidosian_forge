"""Lighting configuration for Panda3D renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from eidosian_core import eidosian

try:
    from panda3d.core import DirectionalLight, AmbientLight, Vec4  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    DirectionalLight = AmbientLight = Vec4 = None


@dataclass(frozen=True)
class LightingConfig:
    """Lighting configuration."""

    ambient_color: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0)
    directional_color: tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0)
    direction: tuple[float, float, float] = (-1.0, -1.0, -2.0)

    def __post_init__(self) -> None:
        if len(self.ambient_color) != 4:
            raise ValueError("ambient_color must be RGBA")
        if len(self.directional_color) != 4:
            raise ValueError("directional_color must be RGBA")
        if len(self.direction) != 3:
            raise ValueError("direction must be XYZ")


@eidosian()
def attach_default_lights(render: Any, config: LightingConfig | None = None) -> None:
    """Attach ambient and directional lights to the scene."""

    if DirectionalLight is None or AmbientLight is None or Vec4 is None:
        raise RuntimeError("Panda3D is not available. Install with 'pip install panda3d'.")
    config = config or LightingConfig()

    ambient = AmbientLight("ambient")
    ambient.set_color(Vec4(*config.ambient_color))
    ambient_np = render.attach_new_node(ambient)
    render.set_light(ambient_np)

    directional = DirectionalLight("directional")
    directional.set_color(Vec4(*config.directional_color))
    directional_np = render.attach_new_node(directional)
    directional_np.set_hpr(0, -45, 0)
    render.set_light(directional_np)
