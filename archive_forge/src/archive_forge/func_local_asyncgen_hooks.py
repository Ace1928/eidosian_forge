import pytest
import types
import sys
import collections.abc
from functools import wraps
import gc
from .conftest import mock_sleep
from .. import (
from .. import _impl
@pytest.fixture
def local_asyncgen_hooks():
    old_hooks = get_asyncgen_hooks()
    yield
    set_asyncgen_hooks(*old_hooks)