import pytest
import types
import sys
import collections.abc
from functools import wraps
import gc
from .conftest import mock_sleep
from .. import (
from .. import _impl
@types.coroutine
def next_me():
    assert (yield 'next me') is None