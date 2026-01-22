import palettable
from palettable.palette import Palette
def test_cmocean():
    assert isinstance(palettable.cmocean.sequential.Amp_8, Palette)
    assert isinstance(palettable.cmocean.diverging.Balance_8, Palette)