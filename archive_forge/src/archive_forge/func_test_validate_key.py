from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_validate_key():
    validate_key(1)
    validate_key(('x', 1))
    with pytest.raises(TypeError, match='Unexpected key type.*list'):
        validate_key(['x', 1])
    with pytest.raises(TypeError, match='unexpected key type at index=1'):
        validate_key((2, int))