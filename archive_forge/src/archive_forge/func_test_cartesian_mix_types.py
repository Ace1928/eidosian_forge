import numpy as np
import pytest
from scipy import linalg, sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.special import expit
from sklearn.datasets import make_low_rank_matrix, make_sparse_spd_matrix
from sklearn.utils import gen_batches
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._testing import (
from sklearn.utils.extmath import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('arrays, output_dtype', [([np.array([1, 2, 3], dtype=np.int32), np.array([4, 5], dtype=np.int64)], np.dtype(np.int64)), ([np.array([1, 2, 3], dtype=np.int32), np.array([4, 5], dtype=np.float64)], np.dtype(np.float64)), ([np.array([1, 2, 3], dtype=np.int32), np.array(['x', 'y'], dtype=object)], np.dtype(object))])
def test_cartesian_mix_types(arrays, output_dtype):
    """Check that the cartesian product works with mixed types."""
    output = cartesian(arrays)
    assert output.dtype == output_dtype