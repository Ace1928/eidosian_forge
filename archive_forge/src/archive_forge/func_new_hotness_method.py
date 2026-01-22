from __future__ import annotations
import inspect
import warnings
import pytest
from .._deprecate import (
from . import module_with_deprecations
def new_hotness_method(self) -> str:
    return 'new hotness method'