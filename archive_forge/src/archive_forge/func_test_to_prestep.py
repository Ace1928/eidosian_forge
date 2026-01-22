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
def test_to_prestep():
    x = np.arange(4)
    y1 = np.arange(4)
    y2 = np.arange(4)[::-1]
    xs, y1s, y2s = cbook.pts_to_prestep(x, y1, y2)
    x_target = np.asarray([0, 0, 1, 1, 2, 2, 3], dtype=float)
    y1_target = np.asarray([0, 1, 1, 2, 2, 3, 3], dtype=float)
    y2_target = np.asarray([3, 2, 2, 1, 1, 0, 0], dtype=float)
    assert_array_equal(x_target, xs)
    assert_array_equal(y1_target, y1s)
    assert_array_equal(y2_target, y2s)
    xs, y1s = cbook.pts_to_prestep(x, y1)
    assert_array_equal(x_target, xs)
    assert_array_equal(y1_target, y1s)