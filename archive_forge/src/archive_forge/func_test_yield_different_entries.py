import pytest
import types
import sys
import collections.abc
from functools import wraps
import gc
from .conftest import mock_sleep
from .. import (
from .. import _impl
def test_yield_different_entries():
    coro = collect(yield_after_different_entries())
    yielded = hostile_coroutine_runner(coro)
    assert yielded == [1, 2, 3, 4]