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
def test_contiguous_regions():
    a, b, c = (3, 4, 5)
    mask = [True] * a + [False] * b + [True] * c
    expected = [(0, a), (a + b, a + b + c)]
    assert cbook.contiguous_regions(mask) == expected
    d, e = (6, 7)
    mask = mask + [False] * e
    assert cbook.contiguous_regions(mask) == expected
    mask = [False] * d + mask[:-e]
    expected = [(d, d + a), (d + a + b, d + a + b + c)]
    assert cbook.contiguous_regions(mask) == expected
    mask = mask + [False] * e
    assert cbook.contiguous_regions(mask) == expected
    assert cbook.contiguous_regions([False] * 5) == []
    assert cbook.contiguous_regions([]) == []