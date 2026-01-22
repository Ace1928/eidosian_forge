from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize('as_dataset', (True, False))
def test_weighted_non_DataArray_weights(as_dataset: bool) -> None:
    data: DataArray | Dataset = DataArray([1, 2])
    if as_dataset:
        data = data.to_dataset(name='data')
    with pytest.raises(ValueError, match='`weights` must be a DataArray'):
        data.weighted([1, 2])