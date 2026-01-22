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
def test_Expr__single_arg():
    p = Pressure1(3)
    assert abs(p({'temperature': 273.15, 'volume': 0.17, 'R': 8.314}) - 3 * 8.314 * 273.15 / 0.17) < 1e-15