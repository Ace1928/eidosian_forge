import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_as_index(self):
    """Test results if `as_index=True`."""
    assert_equal(_as_pairs([2.6, 3.3], 10, as_index=True), np.array([[3, 3]] * 10, dtype=np.intp))
    assert_equal(_as_pairs([2.6, 4.49], 10, as_index=True), np.array([[3, 4]] * 10, dtype=np.intp))
    for x in (-3, [-3], [[-3]], [-3, 4], [3, -4], [[-3, 4]], [[4, -3]], [[1, 2]] * 9 + [[1, -2]]):
        with pytest.raises(ValueError, match='negative values'):
            _as_pairs(x, 10, as_index=True)