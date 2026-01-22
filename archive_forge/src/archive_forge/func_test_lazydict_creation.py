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
def test_lazydict_creation(self):
    lazy_dicts = []
    lazy_dicts += [LazyDict(DATA['data_per_streamline_func'])]
    lazy_dicts += [LazyDict(**DATA['data_per_streamline_func'])]
    expected_keys = DATA['data_per_streamline_func'].keys()
    for data_dict in lazy_dicts:
        assert is_lazy_dict(data_dict)
        assert data_dict.keys() == expected_keys
        for k in data_dict.keys():
            assert_array_equal(list(data_dict[k]), list(DATA['data_per_streamline'][k]))
        assert len(data_dict) == len(DATA['data_per_streamline_func'])