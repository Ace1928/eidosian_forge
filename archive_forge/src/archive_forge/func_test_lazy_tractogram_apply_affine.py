import copy
import operator
import sys
import unittest
import warnings
from collections import defaultdict
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ...testing import assert_arrays_equal, clear_and_catch_warnings
from .. import tractogram as module_tractogram
from ..tractogram import (
def test_lazy_tractogram_apply_affine(self):
    affine = np.eye(4)
    scaling = np.array((1, 2, 3), dtype=float)
    affine[range(3), range(3)] = scaling
    tractogram = DATA['lazy_tractogram'].copy()
    transformed_tractogram = tractogram.apply_affine(affine)
    assert transformed_tractogram is not tractogram
    assert_array_equal(tractogram._affine_to_apply, np.eye(4))
    assert_array_equal(tractogram.affine_to_rasmm, np.eye(4))
    assert_array_equal(transformed_tractogram._affine_to_apply, affine)
    assert_array_equal(transformed_tractogram.affine_to_rasmm, np.dot(np.eye(4), np.linalg.inv(affine)))
    with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
        check_tractogram(transformed_tractogram, streamlines=[s * scaling for s in DATA['streamlines']], data_per_streamline=DATA['data_per_streamline'], data_per_point=DATA['data_per_point'])
    transformed_tractogram = transformed_tractogram.apply_affine(affine)
    assert_array_equal(transformed_tractogram._affine_to_apply, np.dot(affine, affine))
    assert_array_equal(transformed_tractogram.affine_to_rasmm, np.dot(np.eye(4), np.dot(np.linalg.inv(affine), np.linalg.inv(affine))))
    tractogram = DATA['lazy_tractogram'].copy()
    tractogram.affine_to_rasmm = None
    with pytest.raises(ValueError):
        tractogram.to_world()
    tractogram = DATA['lazy_tractogram'].copy()
    tractogram.affine_to_rasmm = None
    transformed_tractogram = tractogram.apply_affine(affine)
    assert_array_equal(transformed_tractogram._affine_to_apply, affine)
    assert transformed_tractogram.affine_to_rasmm is None
    with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
        check_tractogram(transformed_tractogram, streamlines=[s * scaling for s in DATA['streamlines']], data_per_streamline=DATA['data_per_streamline'], data_per_point=DATA['data_per_point'])
    tractogram = DATA['lazy_tractogram'].copy()
    with pytest.raises(ValueError):
        tractogram.apply_affine(affine=np.eye(4), lazy=False)