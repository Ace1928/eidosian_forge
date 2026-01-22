from chempy.util.testing import requires
from chempy.units import units_library, default_units as u
from ..numbers import (
def test_roman():
    assert roman(4) == 'IV'
    assert roman(20) == 'XX'
    assert roman(94) == 'XCIV'
    assert roman(501) == 'DI'