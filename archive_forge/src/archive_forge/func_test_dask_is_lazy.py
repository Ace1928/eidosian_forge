from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
@requires_dask
def test_dask_is_lazy(self) -> None:
    store = InaccessibleVariableDataStore()
    create_test_data().dump_to_store(store)
    ds = open_dataset(store).chunk()
    with pytest.raises(UnexpectedDataAccess):
        ds.load()
    with pytest.raises(UnexpectedDataAccess):
        ds['var1'].values
    ds.var1.data
    ds.isel(time=10)
    ds.isel(time=slice(10), dim1=[0]).isel(dim1=0, dim2=-1)
    ds.transpose()
    ds.mean()
    ds.fillna(0)
    ds.rename({'dim1': 'foobar'})
    ds.set_coords('var1')
    ds.drop_vars('var1')