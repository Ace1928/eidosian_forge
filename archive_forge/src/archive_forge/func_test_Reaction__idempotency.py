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
def test_Reaction__idempotency():
    with pytest.raises(ValueError):
        Reaction({'A': 1}, {'A': 1})
    with pytest.raises(ValueError):
        Reaction({}, {})
    with pytest.raises(ValueError):
        Reaction({'A': 1}, {'B': 1}, inact_reac={'B': 1}, inact_prod={'A': 1})