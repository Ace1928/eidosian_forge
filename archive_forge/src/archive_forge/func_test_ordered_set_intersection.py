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
def test_ordered_set_intersection() -> None:
    assert list(ordered_set_intersection([[1, 2]])) == [1, 2]
    assert list(ordered_set_intersection([[1, 2], [2, 1]])) == [1, 2]
    assert list(ordered_set_intersection([[1, 2], [1, 3]])) == [1]
    assert list(ordered_set_intersection([[1, 2], [2]])) == [2]