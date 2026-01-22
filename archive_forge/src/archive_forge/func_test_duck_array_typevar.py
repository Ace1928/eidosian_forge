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
def test_duck_array_typevar(a: duckarray[Any, _DType]) -> duckarray[Any, _DType]:
    b: duckarray[Any, _DType] = a
    if isinstance(b, _arrayfunction_or_api):
        return b
    else:
        raise TypeError(f'a ({type(a)}) is not a valid _arrayfunction or _arrayapi')