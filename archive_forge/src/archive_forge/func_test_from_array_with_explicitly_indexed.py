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
def test_from_array_with_explicitly_indexed(self, random_inputs: np.ndarray[Any, Any]) -> None:
    array: CustomArray[Any, Any]
    array = CustomArray(random_inputs)
    output: NamedArray[Any, Any]
    output = from_array(('x', 'y', 'z'), array)
    assert isinstance(output.data, np.ndarray)
    array2: CustomArrayIndexable[Any, Any]
    array2 = CustomArrayIndexable(random_inputs)
    output2: NamedArray[Any, Any]
    output2 = from_array(('x', 'y', 'z'), array2)
    assert isinstance(output2.data, CustomArrayIndexable)