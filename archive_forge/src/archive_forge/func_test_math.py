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
def test_math(self) -> None:
    x = self.x
    v = self.v
    a = self.dv
    assert_equal(a, +a)
    assert_equal(a, a + 0)
    assert_equal(a, 0 + a)
    assert_equal(a, a + 0 * v)
    assert_equal(a, 0 * v + a)
    assert_equal(a, a + 0 * x)
    assert_equal(a, 0 * x + a)
    assert_equal(a, a + 0 * a)
    assert_equal(a, 0 * a + a)