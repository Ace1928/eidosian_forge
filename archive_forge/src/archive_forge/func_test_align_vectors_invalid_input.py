import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_align_vectors_invalid_input():
    with pytest.raises(ValueError, match='Expected input `a` to have shape'):
        Rotation.align_vectors([1, 2, 3, 4], [1, 2, 3])
    with pytest.raises(ValueError, match='Expected input `b` to have shape'):
        Rotation.align_vectors([1, 2, 3], [1, 2, 3, 4])
    with pytest.raises(ValueError, match='Expected inputs `a` and `b` to have same shapes'):
        Rotation.align_vectors([[1, 2, 3], [4, 5, 6]], [[1, 2, 3]])
    with pytest.raises(ValueError, match='Expected `weights` to be 1 dimensional'):
        Rotation.align_vectors([[1, 2, 3]], [[1, 2, 3]], weights=[[1]])
    with pytest.raises(ValueError, match='Expected `weights` to have number of values'):
        Rotation.align_vectors([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], weights=[1, 2, 3])
    with pytest.raises(ValueError, match='`weights` may not contain negative values'):
        Rotation.align_vectors([[1, 2, 3]], [[1, 2, 3]], weights=[-1])
    with pytest.raises(ValueError, match='Only one infinite weight is allowed'):
        Rotation.align_vectors([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], weights=[np.inf, np.inf])
    with pytest.raises(ValueError, match='Cannot align zero length primary vectors'):
        Rotation.align_vectors([[0, 0, 0]], [[1, 2, 3]])
    with pytest.raises(ValueError, match='Cannot return sensitivity matrix'):
        Rotation.align_vectors([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], return_sensitivity=True, weights=[np.inf, 1])
    with pytest.raises(ValueError, match='Cannot return sensitivity matrix'):
        Rotation.align_vectors([[1, 2, 3]], [[1, 2, 3]], return_sensitivity=True)