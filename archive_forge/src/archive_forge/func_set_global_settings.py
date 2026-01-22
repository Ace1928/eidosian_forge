from __future__ import annotations
import sys
from typing import Any, Type
import inspect
from contextlib import contextmanager
from functools import cmp_to_key, update_wrapper
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.function import AppliedUndef, UndefinedFunction, Function
@classmethod
def set_global_settings(cls, **settings):
    """Set system-wide printing settings. """
    for key, val in settings.items():
        if val is not None:
            cls._global_settings[key] = val