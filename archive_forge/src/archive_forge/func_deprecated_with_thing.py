from __future__ import annotations
import inspect
import warnings
import pytest
from .._deprecate import (
from . import module_with_deprecations
@deprecated('1.2', thing='the thing', issue=None, instead=None)
def deprecated_with_thing() -> int:
    return 72