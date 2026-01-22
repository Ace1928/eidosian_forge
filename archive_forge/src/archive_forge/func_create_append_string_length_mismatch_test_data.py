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
def create_append_string_length_mismatch_test_data(dtype) -> tuple[Dataset, Dataset]:

    def make_datasets(data, data_to_append) -> tuple[Dataset, Dataset]:
        ds = xr.Dataset({'temperature': (['time'], data)}, coords={'time': [0, 1, 2]})
        ds_to_append = xr.Dataset({'temperature': (['time'], data_to_append)}, coords={'time': [0, 1, 2]})
        assert_writeable(ds)
        assert_writeable(ds_to_append)
        return (ds, ds_to_append)
    u2_strings = ['ab', 'cd', 'ef']
    u5_strings = ['abc', 'def', 'ghijk']
    s2_strings = np.array(['aa', 'bb', 'cc'], dtype='|S2')
    s3_strings = np.array(['aaa', 'bbb', 'ccc'], dtype='|S3')
    if dtype == 'U':
        return make_datasets(u2_strings, u5_strings)
    elif dtype == 'S':
        return make_datasets(s2_strings, s3_strings)
    else:
        raise ValueError(f'unsupported dtype {dtype}.')