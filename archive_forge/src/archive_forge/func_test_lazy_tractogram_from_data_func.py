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
def test_lazy_tractogram_from_data_func(self):
    tractogram = LazyTractogram.from_data_func(lambda: iter([]))
    with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
        check_tractogram(tractogram)
    data = [DATA['streamlines'], DATA['fa'], DATA['colors'], DATA['mean_curvature'], DATA['mean_torsion'], DATA['mean_colors']]

    def _data_gen():
        for d in zip(*data):
            data_for_points = {'fa': d[1], 'colors': d[2]}
            data_for_streamline = {'mean_curvature': d[3], 'mean_torsion': d[4], 'mean_colors': d[5]}
            yield TractogramItem(d[0], data_for_streamline, data_for_points)
    tractogram = LazyTractogram.from_data_func(_data_gen)
    with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
        assert_tractogram_equal(tractogram, DATA['tractogram'])
    with pytest.raises(TypeError):
        LazyTractogram.from_data_func(_data_gen())