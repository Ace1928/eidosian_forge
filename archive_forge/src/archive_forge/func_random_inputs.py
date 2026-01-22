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
@pytest.fixture
def random_inputs(self) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))