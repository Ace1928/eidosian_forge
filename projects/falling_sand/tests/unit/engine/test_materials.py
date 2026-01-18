import pytest

from falling_sand.engine.materials import Material, MaterialPalette


def test_palette_colors() -> None:
    palette = MaterialPalette()
    assert palette.color_for(Material.AIR) == palette.air
    assert palette.color_for(Material.SOLID) == palette.solid


def test_palette_unknown_material() -> None:
    palette = MaterialPalette()
    with pytest.raises(ValueError, match="Unknown material"):
        palette.color_for(99)  # type: ignore[arg-type]
