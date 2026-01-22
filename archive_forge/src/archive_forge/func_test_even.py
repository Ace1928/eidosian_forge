import operator
import typing as t
from collections import abc
from numbers import Number
from .runtime import Undefined
from .utils import pass_environment
def test_even(value: int) -> bool:
    """Return true if the variable is even."""
    return value % 2 == 0