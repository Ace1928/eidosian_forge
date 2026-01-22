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
def sum_add(a, b, core_dims, on_missing_core_dim):
    return apply_ufunc(lambda a, b, axis=None: a.sum(axis) + b.sum(axis), a, b, input_core_dims=core_dims, on_missing_core_dim=on_missing_core_dim)