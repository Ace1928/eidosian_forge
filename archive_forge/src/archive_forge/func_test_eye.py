import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
@pytest.mark.parametrize('eye', [construct.eye, construct.eye_array])
def test_eye(self, eye):
    assert_equal(eye(1, 1).toarray(), [[1]])
    assert_equal(eye(2, 3).toarray(), [[1, 0, 0], [0, 1, 0]])
    assert_equal(eye(3, 2).toarray(), [[1, 0], [0, 1], [0, 0]])
    assert_equal(eye(3, 3).toarray(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert_equal(eye(3, 3, dtype='int16').dtype, np.dtype('int16'))
    for m in [3, 5]:
        for n in [3, 5]:
            for k in range(-5, 6):
                if k > 0 and k > n or (k < 0 and abs(k) > m):
                    with pytest.raises(ValueError, match='Offset.*out of bounds'):
                        eye(m, n, k=k)
                else:
                    assert_equal(eye(m, n, k=k).toarray(), np.eye(m, n, k=k))
                    if m == n:
                        assert_equal(eye(m, k=k).toarray(), np.eye(m, n, k=k))