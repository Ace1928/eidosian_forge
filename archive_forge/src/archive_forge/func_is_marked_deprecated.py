import inspect
import re
import types
import warnings
from functools import wraps
from typing import Any, Callable, TypeVar, Union
def is_marked_deprecated(func):
    """
    Is the function marked as deprecated.
    """
    return getattr(func, _DEPRECATED_MARK_ATTR_NAME, False)