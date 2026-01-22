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
def test_da_groupby_empty() -> None:
    empty_array = xr.DataArray([], dims='dim')
    with pytest.raises(ValueError):
        empty_array.groupby('dim')