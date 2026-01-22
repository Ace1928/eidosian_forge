from __future__ import annotations
import functools
import operator
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, duck_array_ops
from xarray.tests import (
from xarray.tests.test_plot import PlotTestCase
from xarray.tests.test_variable import _PAD_XR_NP_ARGS
def extract_units(obj):
    if isinstance(obj, xr.Dataset):
        vars_units = {name: array_extract_units(value) for name, value in obj.data_vars.items()}
        coords_units = {name: array_extract_units(value) for name, value in obj.coords.items()}
        units = {**vars_units, **coords_units}
    elif isinstance(obj, xr.DataArray):
        vars_units = {obj.name: array_extract_units(obj)}
        coords_units = {name: array_extract_units(value) for name, value in obj.coords.items()}
        units = {**vars_units, **coords_units}
    elif isinstance(obj, xr.Variable):
        vars_units = {None: array_extract_units(obj.data)}
        units = {**vars_units}
    elif isinstance(obj, Quantity):
        vars_units = {None: array_extract_units(obj)}
        units = {**vars_units}
    else:
        units = {}
    return units