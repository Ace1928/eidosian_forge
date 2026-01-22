from __future__ import annotations
import sys
import warnings
from functools import wraps
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar, TypeVar
import attrs
def enable_attribute_deprecations(module_name: str) -> None:
    module = sys.modules[module_name]
    module.__class__ = _ModuleWithDeprecations
    assert isinstance(module, _ModuleWithDeprecations)
    module.__deprecated_attributes__ = {}