"""Material definitions for the falling sand engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from eidosian_core import eidosian


class Material(IntEnum):
    """Discrete material identifiers for simulation."""

    AIR = 0
    SOLID = 1
    GRANULAR = 2
    LIQUID = 3
    GAS = 4


@dataclass(frozen=True)
class MaterialPalette:
    """Color palette for rendering materials."""

    air: tuple[float, float, float] = (0.0, 0.0, 0.0)
    solid: tuple[float, float, float] = (0.6, 0.6, 0.6)
    granular: tuple[float, float, float] = (0.9, 0.8, 0.4)
    liquid: tuple[float, float, float] = (0.2, 0.4, 0.9)
    gas: tuple[float, float, float] = (0.7, 0.9, 0.9)
    solid_alpha: float = 1.0
    granular_alpha: float = 0.75
    liquid_alpha: float = 0.6
    gas_alpha: float = 0.35

    @eidosian()
    def color_for(self, material: Material) -> tuple[float, float, float]:
        """Return RGB color for a material."""

        if material == Material.AIR:
            return self.air
        if material == Material.SOLID:
            return self.solid
        if material == Material.GRANULAR:
            return self.granular
        if material == Material.LIQUID:
            return self.liquid
        if material == Material.GAS:
            return self.gas
        raise ValueError(f"Unknown material: {material}")

    @eidosian()
    def rgba_for(self, material: Material) -> tuple[float, float, float, float]:
        """Return RGBA color for a material."""

        if material == Material.SOLID:
            alpha = self.solid_alpha
        elif material == Material.GRANULAR:
            alpha = self.granular_alpha
        elif material == Material.LIQUID:
            alpha = self.liquid_alpha
        elif material == Material.GAS:
            alpha = self.gas_alpha
        elif material == Material.AIR:
            alpha = 0.0
        else:
            raise ValueError(f"Unknown material: {material}")
        rgb = self.color_for(material) if material != Material.AIR else self.air
        return (rgb[0], rgb[1], rgb[2], alpha)


MOVES_DOWN = (Material.GRANULAR, Material.LIQUID)
MOVES_UP = (Material.GAS,)
IMMOVABLE = (Material.SOLID,)
