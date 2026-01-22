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
def test_groupby_iter(self) -> None:
    for (act_x, act_dv), (exp_x, exp_ds) in zip(self.dv.groupby('y', squeeze=False), self.ds.groupby('y', squeeze=False)):
        assert exp_x == act_x
        assert_identical(exp_ds['foo'], act_dv)
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        for (_, exp_dv), (_, act_dv) in zip(self.dv.groupby('x'), self.dv.groupby('x')):
            assert_identical(exp_dv, act_dv)