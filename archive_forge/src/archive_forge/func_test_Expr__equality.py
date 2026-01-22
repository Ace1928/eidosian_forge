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
def test_Expr__equality():
    K1 = MyK(unique_keys=('H1', 'S1'))
    K2 = MyK(unique_keys=('H2', 'S2'))
    assert K1 != K2
    assert K1 == K1
    K3 = MyK([23000.0 * u.J / u.mol, 42 * u.J / u.mol / u.K])
    K4 = MyK([23000.0, 42])
    K5 = MyK([24000.0 * u.J / u.mol, 42 * u.J / u.mol / u.K])
    K6 = MyK([23000.0, 43])
    assert K3 == K3
    assert K4 == K4
    assert K5 != K3
    assert K4 != K6
    assert K3 != K4