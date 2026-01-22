import palettable
from palettable.palette import Palette
def test_tableau():
    assert isinstance(palettable.tableau.ColorBlind_10, Palette)