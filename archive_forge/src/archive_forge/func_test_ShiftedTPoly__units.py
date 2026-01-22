import math
import pytest
from chempy import Reaction
from chempy.units import (
from chempy.util._expr import Log10, Constant
from chempy.util.testing import requires
from ..rates import MassAction, Radiolytic
from .._rates import (
@requires(units_library)
def test_ShiftedTPoly__units():
    stp1 = ShiftedTPoly([273.15 * u.kelvin, 5, 7 / u.kelvin])
    allclose(stp1({'temperature': 274.15 * u.kelvin}), 5 + 7)
    allclose(stp1({'temperature': 273.15 * u.kelvin}), 5)
    stp2 = ShiftedTPoly([273.15 * u.kelvin, 5 * u.m, 7 * u.m / u.kelvin, 13 * u.m * u.kelvin ** (-2)])
    allclose(stp2({'temperature': 274.15 * u.kelvin}), (5 + 7 + 13) * u.m)
    allclose(stp2({'temperature': 273.15 * u.kelvin}), 5 * u.m)