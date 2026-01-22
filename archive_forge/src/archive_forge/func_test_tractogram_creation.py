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
def test_tractogram_creation(self):
    tractogram = Tractogram()
    check_tractogram(tractogram)
    assert tractogram.affine_to_rasmm is None
    tractogram = Tractogram(streamlines=DATA['streamlines'])
    check_tractogram(tractogram, DATA['streamlines'])
    affine = np.diag([1, 2, 3, 1])
    tractogram = Tractogram(affine_to_rasmm=affine)
    assert_array_equal(tractogram.affine_to_rasmm, affine)
    tractogram = Tractogram(DATA['streamlines'], DATA['data_per_streamline'], DATA['data_per_point'])
    check_tractogram(tractogram, DATA['streamlines'], DATA['data_per_streamline'], DATA['data_per_point'])
    assert is_data_dict(tractogram.data_per_streamline)
    assert is_data_dict(tractogram.data_per_point)
    tractogram2 = Tractogram(tractogram.streamlines, tractogram.data_per_streamline, tractogram.data_per_point)
    assert_tractogram_equal(tractogram2, tractogram)
    tractogram = LazyTractogram(DATA['streamlines_func'], DATA['data_per_streamline_func'], DATA['data_per_point_func'])
    tractogram2 = Tractogram(tractogram.streamlines, tractogram.data_per_streamline, tractogram.data_per_point)
    wrong_data = [[(1, 0, 0)] * 1, [(0, 1, 0), (0, 1)], [(0, 0, 1)] * 5]
    data_per_point = {'wrong_data': wrong_data}
    with pytest.raises(ValueError):
        Tractogram(streamlines=DATA['streamlines'], data_per_point=data_per_point)
    wrong_data = [[(1, 0, 0)] * 1, [(0, 1)] * 2, [(0, 0, 1)] * 5]
    data_per_point = {'wrong_data': wrong_data}
    with pytest.raises(ValueError):
        Tractogram(streamlines=DATA['streamlines'], data_per_point=data_per_point)