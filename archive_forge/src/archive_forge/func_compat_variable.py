import functools
import warnings
from collections.abc import Hashable
from typing import Union
import numpy as np
import pandas as pd
from xarray.core import duck_array_ops, formatting, utils
from xarray.core.coordinates import Coordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import Index, PandasIndex, PandasMultiIndex, default_indexes
from xarray.core.variable import IndexVariable, Variable
def compat_variable(a, b):
    a = getattr(a, 'variable', a)
    b = getattr(b, 'variable', b)
    return a.dims == b.dims and (a._data is b._data or equiv(a.data, b.data))