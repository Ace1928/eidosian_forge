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
def test_create_Poly():
    PolyT = create_Poly('T')
    p = PolyT([1, 2, 3, 4, 5])
    assert p({'T': 11}) == 1 + 2 * 11 + 3 * 11 ** 2 + 4 * 11 ** 3 + 5 * 11 ** 4