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
def test_lazy_tractogram_extend(self):
    t = DATA['lazy_tractogram'].copy()
    new_t = DATA['lazy_tractogram'].copy()
    for op in (operator.add, operator.iadd, extender):
        with pytest.raises(NotImplementedError):
            op(new_t, t)