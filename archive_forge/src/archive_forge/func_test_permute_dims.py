from __future__ import annotations
import copy
import warnings
from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, cast, overload
import numpy as np
import pytest
from xarray.core.indexing import ExplicitlyIndexed
from xarray.namedarray._typing import (
from xarray.namedarray.core import NamedArray, from_array
@pytest.mark.parametrize('dims, expected_sizes', [((), {'y': 5, 'x': 2}), (['y', 'x'], {'y': 5, 'x': 2}), (['y', ...], {'y': 5, 'x': 2})])
def test_permute_dims(self, target: NamedArray[Any, np.dtype[np.float32]], dims: _DimsLike, expected_sizes: dict[_Dim, _IntOrUnknown]) -> None:
    actual = target.permute_dims(*dims)
    assert actual.sizes == expected_sizes