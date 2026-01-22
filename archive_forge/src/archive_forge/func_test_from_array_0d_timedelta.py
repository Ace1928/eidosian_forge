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
@pytest.mark.parametrize('timedelta, expected_dtype', [(np.timedelta64(1, 'D'), np.dtype('timedelta64[D]')), (np.timedelta64(1, 's'), np.dtype('timedelta64[s]')), (np.timedelta64(1, 'm'), np.dtype('timedelta64[m]')), (np.timedelta64(1, 'h'), np.dtype('timedelta64[h]')), (np.timedelta64(1, 'us'), np.dtype('timedelta64[us]')), (np.timedelta64(1, 'ns'), np.dtype('timedelta64[ns]')), (np.timedelta64(1, 'ps'), np.dtype('timedelta64[ps]')), (np.timedelta64(1, 'fs'), np.dtype('timedelta64[fs]')), (np.timedelta64(1, 'as'), np.dtype('timedelta64[as]'))])
def test_from_array_0d_timedelta(self, timedelta: np.timedelta64, expected_dtype: np.dtype[np.timedelta64]) -> None:
    named_array: NamedArray[Any, Any]
    named_array = from_array([], timedelta)
    assert named_array.dtype == expected_dtype
    assert named_array.data == timedelta