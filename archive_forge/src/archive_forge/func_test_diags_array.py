import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def test_diags_array():
    """Tests of diags_array that do not rely on diags wrapper."""
    diag = np.arange(1, 5)
    assert_array_equal(construct.diags_array(diag).toarray(), np.diag(diag))
    assert_array_equal(construct.diags_array(diag, offsets=2).toarray(), np.diag(diag, k=2))
    assert_array_equal(construct.diags_array(diag, offsets=2, shape=(4, 4)).toarray(), np.diag(diag, k=2)[:4, :4])
    with pytest.raises(ValueError, match='.*out of bounds'):
        construct.diags(np.arange(1, 5), 5, shape=(4, 4))