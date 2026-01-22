from __future__ import annotations
import itertools
import pickle
from typing import Any
from unittest.mock import patch, Mock
from datetime import datetime, date, timedelta
import numpy as np
from numpy.testing import (assert_array_equal, assert_approx_equal,
import pytest
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
from matplotlib.cbook import delete_masked_points, strip_math
def test_rgba(self):
    a_masked = np.ma.array([1, 2, 3, np.nan, np.nan, 6], mask=[False, False, True, True, False, False])
    a_rgba = mcolors.to_rgba_array(['r', 'g', 'b', 'c', 'm', 'y'])
    actual = delete_masked_points(a_masked, a_rgba)
    ind = [0, 1, 5]
    assert_array_equal(actual[0], a_masked[ind].compressed())
    assert_array_equal(actual[1], a_rgba[ind])