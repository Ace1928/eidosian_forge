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
def make_groupby_multidim_example_array(self) -> DataArray:
    return DataArray([[[0, 1], [2, 3]], [[5, 10], [15, 20]]], coords={'lon': (['ny', 'nx'], [[30, 40], [40, 50]]), 'lat': (['ny', 'nx'], [[10, 10], [20, 20]])}, dims=['time', 'ny', 'nx'])