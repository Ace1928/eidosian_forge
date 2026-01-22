from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import deepcopy
from textwrap import dedent
from typing import Any, Final, Literal, cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import (
from xarray.coding.times import CFDatetimeCoder
from xarray.core import dtypes
from xarray.core.common import full_like
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import Index, PandasIndex, filter_indexes_from_coords
from xarray.core.types import QueryEngineOptions, QueryParserOptions
from xarray.core.utils import is_scalar
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_reindex_empty_array_dtype(self) -> None:
    x = xr.DataArray([], dims=('x',), coords={'x': []}).astype('float32')
    y = x.reindex(x=[1.0, 2.0])
    assert x.dtype == y.dtype, 'Dtype of reindexed DataArray should match dtype of the original DataArray'
    assert y.dtype == np.float32, 'Dtype of reindexed DataArray should remain float32'