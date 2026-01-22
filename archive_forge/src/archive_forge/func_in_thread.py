import pytest
import types
import sys
import collections.abc
from functools import wraps
import gc
from .conftest import mock_sleep
from .. import (
from .. import _impl
def in_thread(results):
    results.append(get_asyncgen_hooks())
    set_asyncgen_hooks(two, one)
    results.append(get_asyncgen_hooks())