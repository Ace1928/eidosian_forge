from __future__ import annotations
import atexit
import concurrent.futures
import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, overload
from langsmith import client as ls_client
from langsmith import run_helpers as rh
from langsmith import utils as ls_utils
def to_be_approximately(self, value: float, precision: int=2) -> None:
    """Assert that the expectation value is approximately equal to the given value.

        Args:
            value: The value to compare against.
            precision: The number of decimal places to round to for comparison.

        Raises:
            AssertionError: If the rounded expectation value
                does not equal the rounded given value.
        """
    self._assert(round(self.value, precision) == round(value, precision), f'Expected {self.key} to be approximately {value}, but got {self.value}', 'to_be_approximately')