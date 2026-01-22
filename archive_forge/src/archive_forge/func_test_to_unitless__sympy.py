from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library, 'sympy')
def test_to_unitless__sympy():
    import sympy as sp
    assert sp.cos(to_unitless(sp.pi)) == -1
    with pytest.raises(Exception):
        to_unitless(sp.pi, u.second)