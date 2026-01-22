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
def test_lazy_tractogram_len(self):
    modules = [module_tractogram]
    with clear_and_catch_warnings(record=True, modules=modules) as w:
        warnings.simplefilter('always')
        tractogram = LazyTractogram(DATA['streamlines_func'])
        assert tractogram._nb_streamlines is None
        assert len(tractogram) == len(DATA['streamlines'])
        assert tractogram._nb_streamlines == len(DATA['streamlines'])
        assert len(w) == 1
        tractogram = LazyTractogram(DATA['streamlines_func'])
        assert len(tractogram) == len(DATA['streamlines'])
        assert len(w) == 2
        assert issubclass(w[-1].category, Warning) is True
        assert len(tractogram) == len(DATA['streamlines'])
        assert len(w) == 2
    with clear_and_catch_warnings(record=True, modules=modules) as w:
        tractogram = LazyTractogram(DATA['streamlines_func'])
        assert tractogram._nb_streamlines is None
        [t for t in tractogram]
        assert tractogram._nb_streamlines == len(DATA['streamlines'])
        assert len(tractogram) == len(DATA['streamlines'])
        assert len(w) == 0