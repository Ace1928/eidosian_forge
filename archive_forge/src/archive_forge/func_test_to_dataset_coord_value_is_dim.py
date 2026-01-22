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
def test_to_dataset_coord_value_is_dim(self) -> None:
    array = DataArray(np.zeros((3, 3)), coords={'x': ['a', 'b', 'c'], 'a': [1, 2, 3]})
    with pytest.raises(ValueError, match=re.escape("dimension 'x' would produce the variables ('a',)") + '.*' + re.escape('DataArray.rename(a=...) or DataArray.assign_coords(x=...)')):
        array.to_dataset('x')
    array2 = DataArray(np.zeros((3, 3, 2)), coords={'x': ['a', 'b', 'c'], 'a': [1, 2, 3], 'b': [0.0, 0.1]})
    with pytest.raises(ValueError, match=re.escape("dimension 'x' would produce the variables ('a', 'b')") + '.*' + re.escape('DataArray.rename(a=..., b=...) or DataArray.assign_coords(x=...)')):
        array2.to_dataset('x')