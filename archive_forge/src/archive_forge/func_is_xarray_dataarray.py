import sys
from collections.abc import Hashable
from functools import wraps
from packaging.version import Version
from types import FunctionType
import bokeh
import numpy as np
import pandas as pd
import param
import holoviews as hv
def is_xarray_dataarray(data):
    if not check_library(data, 'xarray'):
        return False
    from xarray import DataArray
    return isinstance(data, DataArray)