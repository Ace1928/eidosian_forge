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
def test_tractogram_to_world(self):
    tractogram = DATA['lazy_tractogram'].copy()
    affine = np.random.RandomState(1234).randn(4, 4)
    affine[-1] = [0, 0, 0, 1]
    transformed_tractogram = tractogram.apply_affine(affine)
    assert_array_equal(transformed_tractogram.affine_to_rasmm, np.linalg.inv(affine))
    tractogram_world = transformed_tractogram.to_world()
    assert tractogram_world is not transformed_tractogram
    assert_array_almost_equal(tractogram_world.affine_to_rasmm, np.eye(4))
    for s1, s2 in zip(tractogram_world.streamlines, DATA['streamlines']):
        assert_array_almost_equal(s1, s2)
    tractogram_world = tractogram_world.to_world()
    assert_array_almost_equal(tractogram_world.affine_to_rasmm, np.eye(4))
    for s1, s2 in zip(tractogram_world.streamlines, DATA['streamlines']):
        assert_array_almost_equal(s1, s2)
    tractogram = DATA['lazy_tractogram'].copy()
    tractogram.affine_to_rasmm = None
    with pytest.raises(ValueError):
        tractogram.to_world()