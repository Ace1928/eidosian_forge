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
def test_Equilibrium__cancel():
    e1 = Equilibrium({'A': 26, 'B': 20, 'C': 7}, {'D': 4, 'E': 7})
    e2 = Equilibrium({'A': 13, 'B': 3}, {'D': 2})
    coeff = e1.cancel(e2)
    assert coeff == -2