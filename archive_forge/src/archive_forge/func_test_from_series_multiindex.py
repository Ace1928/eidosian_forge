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
def test_from_series_multiindex(self) -> None:
    df = pd.DataFrame({'B': [1, 2, 3], 'A': [4, 5, 6]})
    df = df.rename_axis('num').rename_axis('alpha', axis=1)
    actual = df.stack('alpha').to_xarray()
    assert (actual.sel(alpha='B') == [1, 2, 3]).all()
    assert (actual.sel(alpha='A') == [4, 5, 6]).all()