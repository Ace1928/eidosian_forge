from __future__ import annotations
import inspect
import warnings
import pytest
from .._deprecate import (
from . import module_with_deprecations
def deprecated_thing() -> None:
    warn_deprecated('ice', '1.2', issue=1, instead='water')