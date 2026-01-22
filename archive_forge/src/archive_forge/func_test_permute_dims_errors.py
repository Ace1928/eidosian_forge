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
def test_permute_dims_errors(self, target: NamedArray[Any, np.dtype[np.float32]]) -> None:
    with pytest.raises(ValueError, match="'y'.*permuted list"):
        dims = ['y']
        target.permute_dims(*dims)