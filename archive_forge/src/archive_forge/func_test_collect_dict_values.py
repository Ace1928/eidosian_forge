from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
def test_collect_dict_values() -> None:
    dicts = [{'x': 1, 'y': 2, 'z': 3}, {'z': 4}, 5]
    expected = [[1, 0, 5], [2, 0, 5], [3, 4, 5]]
    collected = collect_dict_values(dicts, ['x', 'y', 'z'], fill_value=0)
    assert collected == expected