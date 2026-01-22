from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Union
import pytest
from xarray import DataArray, Dataset, Variable
@parametrize_dim
def test_hashable_dims(dim: DimT) -> None:
    v = Variable([dim], [1, 2, 3])
    da = DataArray([1, 2, 3], dims=[dim])
    Dataset({'a': ([dim], [1, 2, 3])})
    DataArray(v)
    Dataset({'a': v})
    Dataset({'a': da})