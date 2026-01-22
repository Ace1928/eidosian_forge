from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test_get_physical_dimensionality():
    assert get_physical_dimensionality(3 * u.mole) == {'amount': 1}
    assert get_physical_dimensionality([3 * u.mole]) == {'amount': 1}
    assert get_physical_dimensionality(42) == {}