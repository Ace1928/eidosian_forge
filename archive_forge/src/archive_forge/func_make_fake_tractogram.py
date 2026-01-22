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
def make_fake_tractogram(list_nb_points, data_per_point_shapes={}, data_for_streamline_shapes={}, rng=None):
    """Make multiple streamlines according to provided requirements."""
    all_streamlines = []
    all_data_per_point = defaultdict(lambda: [])
    all_data_per_streamline = defaultdict(lambda: [])
    for nb_points in list_nb_points:
        data = make_fake_streamline(nb_points, data_per_point_shapes, data_for_streamline_shapes, rng)
        streamline, data_per_point, data_for_streamline = data
        all_streamlines.append(streamline)
        for k, v in data_per_point.items():
            all_data_per_point[k].append(v)
        for k, v in data_for_streamline.items():
            all_data_per_streamline[k].append(v)
    return (all_streamlines, all_data_per_point, all_data_per_streamline)