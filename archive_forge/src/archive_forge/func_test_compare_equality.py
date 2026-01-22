from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test_compare_equality():
    assert compare_equality(3 * u.m, 3 * u.m)
    assert compare_equality(3 * u.m, 0.003 * u.km)
    assert compare_equality(3000.0 * u.mm, 3 * u.m)
    assert not compare_equality(3 * u.m, 2 * u.m)
    assert not compare_equality(3 * u.m, 3 * u.s)
    assert not compare_equality(3 * u.m, 3 * u.m ** 2)
    assert not compare_equality(3 * u.m, np.array(3))
    assert not compare_equality(np.array(3), 3 * u.m)
    assert compare_equality([3, None], [3, None])
    assert not compare_equality([3, None, 3], [3, None, None])
    assert not compare_equality([None, None, 3], [None, None, 2])
    assert compare_equality([3 * u.m, None], [3, None])
    assert not compare_equality([3 * u.m, None], [3 * u.km, None])