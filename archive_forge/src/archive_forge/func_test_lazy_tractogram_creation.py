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
def test_lazy_tractogram_creation(self):
    with pytest.raises(TypeError):
        LazyTractogram(streamlines=DATA['streamlines'])
    streamlines = (x for x in DATA['streamlines'])
    data_per_point = {'colors': (x for x in DATA['colors'])}
    data_per_streamline = {'torsion': (x for x in DATA['mean_torsion']), 'colors': (x for x in DATA['mean_colors'])}
    with pytest.raises(TypeError):
        LazyTractogram(streamlines=streamlines)
    with pytest.raises(TypeError):
        LazyTractogram(data_per_point={'none': None})
    with pytest.raises(TypeError):
        LazyTractogram(data_per_streamline=data_per_streamline)
    with pytest.raises(TypeError):
        LazyTractogram(streamlines=DATA['streamlines'], data_per_point=data_per_point)
    tractogram = LazyTractogram()
    with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
        check_tractogram(tractogram)
    assert tractogram.affine_to_rasmm is None
    tractogram = LazyTractogram(DATA['streamlines_func'], DATA['data_per_streamline_func'], DATA['data_per_point_func'])
    assert is_lazy_dict(tractogram.data_per_streamline)
    assert is_lazy_dict(tractogram.data_per_point)
    [t for t in tractogram]
    assert len(tractogram) == len(DATA['streamlines'])
    for i in range(2):
        assert_tractogram_equal(tractogram, DATA['tractogram'])