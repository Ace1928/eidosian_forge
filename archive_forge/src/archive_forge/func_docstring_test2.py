from __future__ import annotations
import inspect
import warnings
import pytest
from .._deprecate import (
from . import module_with_deprecations
@deprecated('2.1', issue=None, instead='hi')
def docstring_test2() -> None:
    """Hello!"""