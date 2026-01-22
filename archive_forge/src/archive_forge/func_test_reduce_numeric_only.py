from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
def test_reduce_numeric_only(dataset) -> None:
    gb = dataset.groupby('x', squeeze=False)
    with xr.set_options(use_flox=False):
        expected = gb.sum()
    with xr.set_options(use_flox=True):
        actual = gb.sum()
    assert_identical(expected, actual)