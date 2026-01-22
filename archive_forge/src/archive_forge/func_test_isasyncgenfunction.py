import pytest
import types
import sys
import collections.abc
from functools import wraps
import gc
from .conftest import mock_sleep
from .. import (
from .. import _impl
def test_isasyncgenfunction():
    assert isasyncgenfunction(async_range)
    assert not isasyncgenfunction(list)
    assert not isasyncgenfunction(async_range(10))
    if sys.version_info >= (3, 6):
        assert isasyncgenfunction(native_async_range)
        assert not isasyncgenfunction(native_async_range(10))