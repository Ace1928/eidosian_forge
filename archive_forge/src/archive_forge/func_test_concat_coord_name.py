from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable, concat
from xarray.core import dtypes, merge
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import PandasIndex
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
def test_concat_coord_name(self) -> None:
    da = DataArray([0], dims='a')
    da_concat = concat([da, da], dim=DataArray([0, 1], dims='b'))
    assert list(da_concat.coords) == ['b']
    da_concat_std = concat([da, da], dim=DataArray([0, 1]))
    assert list(da_concat_std.coords) == ['dim_0']