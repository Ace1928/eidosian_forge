import math
import pytest
from chempy import Reaction
from chempy.units import (
from chempy.util._expr import Log10, Constant
from chempy.util.testing import requires
from ..rates import MassAction, Radiolytic
from .._rates import (
@requires(units_library)
def test_RTPolyMassAction__units():
    Mps = u.molar / u.second
    kunit = 1 / u.molar ** 2 / u.second
    p = MassAction(ShiftedRTPoly([273.15 * u.K, 7 * kunit, 0.2 * kunit * u.K, 0.03 * kunit * u.K ** 2, 0.004 * kunit * u.K ** 3]))
    r = Reaction({'A': 2, 'B': 1}, {'C': 1}, None, {'B': 1})
    res = p({'A': 11 * u.molar, 'B': 13 * u.molar, 'temperature': 298.15 * u.K}, reaction=r)
    ref = 7 + 0.2 / 25 + 0.03 / 25 ** 2 + 0.004 / 25 ** 3
    assert abs(res - ref * 13 * 11 ** 2 * Mps) < 1e-15