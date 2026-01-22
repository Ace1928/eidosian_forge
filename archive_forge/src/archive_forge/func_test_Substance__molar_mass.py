from functools import reduce
from operator import attrgetter, add
import sys
from sympy import nsimplify
import pytest
from ..util.arithmeticdict import ArithmeticDict
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, to_unitless, allclose
from ..chemistry import (
@requires(parsing_library, units_library)
def test_Substance__molar_mass():
    mw_water = Substance.from_formula('H2O').molar_mass(default_units)
    q = mw_water / ((15.9994 + 2 * 1.008) * default_units.gram / default_units.mol)
    assert abs(q - 1) < 0.001