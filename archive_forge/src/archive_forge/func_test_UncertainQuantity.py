from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@pytest.mark.xfail
@requires(units_library)
def test_UncertainQuantity():
    a = UncertainQuantity([1, 2], u.m, [0.1, 0.2])
    assert a[1] == [2.0] * u.m
    assert (-a)[0] == [-1.0] * u.m
    assert (-a).uncertainty[0] == [0.1] * u.m
    assert (-a)[0] == (a * -1)[0]
    assert (-a).uncertainty[0] == (a * -1).uncertainty[0]
    assert allclose(a, [1, 2] * u.m)