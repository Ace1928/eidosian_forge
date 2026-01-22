import palettable
from palettable.palette import Palette
def test_cartocolors():
    assert isinstance(palettable.cartocolors.sequential.Mint_7, Palette)
    assert isinstance(palettable.cartocolors.diverging.Earth_7, Palette)