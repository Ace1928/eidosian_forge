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
def test_astype_subok(self) -> None:

    class NdArraySubclass(np.ndarray):
        pass
    original = DataArray(NdArraySubclass(np.arange(3)))
    converted_not_subok = original.astype('d', subok=False)
    converted_subok = original.astype('d', subok=True)
    if not isinstance(original.data, NdArraySubclass):
        pytest.xfail('DataArray cannot be backed yet by a subclasses of np.ndarray')
    assert isinstance(converted_not_subok.data, np.ndarray)
    assert not isinstance(converted_not_subok.data, NdArraySubclass)
    assert isinstance(converted_subok.data, NdArraySubclass)