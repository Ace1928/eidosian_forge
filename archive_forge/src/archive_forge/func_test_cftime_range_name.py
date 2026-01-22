from __future__ import annotations
from itertools import product
from typing import Callable, Literal
import numpy as np
import pandas as pd
import pytest
from xarray import CFTimeIndex
from xarray.coding.cftime_offsets import (
from xarray.coding.frequencies import infer_freq
from xarray.core.dataarray import DataArray
from xarray.tests import (
def test_cftime_range_name():
    result = cftime_range(start='2000', periods=4, name='foo')
    assert result.name == 'foo'
    result = cftime_range(start='2000', periods=4)
    assert result.name is None