import math
from operator import add
from functools import reduce
import pytest
from chempy import Substance
from chempy.units import (
from ..testing import requires
from ..pyutil import defaultkeydict
from .._expr import (
from ..parsing import parsing_library
@requires(units_library)
def test_Expr__single_arg__units():
    p = Pressure1(3 * u.mol)
    variables = {'temperature': 273.15 * u.kelvin, 'volume': 170 * u.dm3, 'R': 8.314 * u.J / u.K / u.mol}
    assert allclose(p(variables), 3 * 8.314 * 273.15 / 0.17 * u.Pa)