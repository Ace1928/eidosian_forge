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
def test_tractogram_apply_affine(self):
    tractogram = DATA['tractogram'].copy()
    affine = np.eye(4)
    scaling = np.array((1, 2, 3), dtype=float)
    affine[range(3), range(3)] = scaling
    transformed_tractogram = tractogram.apply_affine(affine, lazy=True)
    assert type(transformed_tractogram) is LazyTractogram
    check_tractogram(transformed_tractogram, streamlines=[s * scaling for s in DATA['streamlines']], data_per_streamline=DATA['data_per_streamline'], data_per_point=DATA['data_per_point'])
    assert_array_equal(transformed_tractogram.affine_to_rasmm, np.dot(np.eye(4), np.linalg.inv(affine)))
    assert_arrays_equal(tractogram.streamlines, DATA['streamlines'])
    transformed_tractogram = tractogram.apply_affine(affine)
    assert transformed_tractogram is tractogram
    check_tractogram(tractogram, streamlines=[s * scaling for s in DATA['streamlines']], data_per_streamline=DATA['data_per_streamline'], data_per_point=DATA['data_per_point'])
    transformed_tractogram = tractogram.apply_affine(affine)
    assert_array_equal(transformed_tractogram.affine_to_rasmm, np.dot(np.eye(4), np.dot(np.linalg.inv(affine), np.linalg.inv(affine))))
    tractogram = DATA['tractogram'].copy()
    transformed_tractogram = tractogram[::2].apply_affine(affine)
    assert transformed_tractogram is not tractogram
    check_tractogram(tractogram[::2], streamlines=[s * scaling for s in DATA['streamlines'][::2]], data_per_streamline=DATA['tractogram'].data_per_streamline[::2], data_per_point=DATA['tractogram'].data_per_point[::2])
    check_tractogram(tractogram[1::2], streamlines=DATA['streamlines'][1::2], data_per_streamline=DATA['tractogram'].data_per_streamline[1::2], data_per_point=DATA['tractogram'].data_per_point[1::2])
    tractogram = DATA['tractogram'].copy()
    affine = np.random.RandomState(1234).randn(4, 4)
    affine[-1] = [0, 0, 0, 1]
    tractogram.apply_affine(affine)
    tractogram.apply_affine(np.linalg.inv(affine))
    assert_array_almost_equal(tractogram.affine_to_rasmm, np.eye(4))
    for s1, s2 in zip(tractogram.streamlines, DATA['streamlines']):
        assert_array_almost_equal(s1, s2)
    tractogram = DATA['tractogram'].copy()
    tractogram.apply_affine(np.eye(4))
    for s1, s2 in zip(tractogram.streamlines, DATA['streamlines']):
        assert_array_almost_equal(s1, s2)
    tractogram = DATA['tractogram'].copy()
    tractogram.affine_to_rasmm = None
    tractogram.apply_affine(affine)
    assert tractogram.affine_to_rasmm is None