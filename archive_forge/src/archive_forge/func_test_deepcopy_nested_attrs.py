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
def test_deepcopy_nested_attrs() -> None:
    """Check attrs deep copy, see :issue:`2835`"""
    da1 = xr.DataArray([[1, 2], [3, 4]], dims=('x', 'y'), coords={'x': [10, 20]})
    da1.attrs['flat'] = '0'
    da1.attrs['nested'] = {'level1a': '1', 'level1b': '1'}
    da2 = da1.copy(deep=True)
    da2.attrs['new'] = '2'
    da2.attrs.update({'new2': '2'})
    da2.attrs['flat'] = '2'
    da2.attrs['nested']['level1a'] = '2'
    da2.attrs['nested'].update({'level1b': '2'})
    assert not da1.identical(da2)
    assert da1.attrs['flat'] != da2.attrs['flat']
    assert da1.attrs['nested'] != da2.attrs['nested']
    assert 'new' not in da1.attrs
    assert 'new2' not in da1.attrs