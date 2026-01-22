import operator
import typing as t
from collections import abc
from numbers import Number
from .runtime import Undefined
from .utils import pass_environment
def test_escaped(value: t.Any) -> bool:
    """Check if the value is escaped."""
    return hasattr(value, '__html__')