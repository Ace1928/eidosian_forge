import sys
import re
import functools
import os
import contextlib
import warnings
import inspect
import pathlib
from typing import Any, Callable
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.utilities.exceptions import ignore_warnings # noqa:F401
def skip_under_pyodide(message):
    """Decorator to skip a test if running under pyodide."""

    def decorator(test_func):

        @functools.wraps(test_func)
        def test_wrapper():
            if _running_under_pyodide():
                skip(message)
            return test_func()
        return test_wrapper
    return decorator