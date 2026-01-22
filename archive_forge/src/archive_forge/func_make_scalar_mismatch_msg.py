import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
def make_scalar_mismatch_msg(actual: Union[bool, int, float, complex], expected: Union[bool, int, float, complex], *, rtol: float, atol: float, identifier: Optional[Union[str, Callable[[str], str]]]=None) -> str:
    """Makes a mismatch error message for scalars.

    Args:
        actual (Union[bool, int, float, complex]): Actual scalar.
        expected (Union[bool, int, float, complex]): Expected scalar.
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.
        identifier (Optional[Union[str, Callable[[str], str]]]): Optional description for the scalars. Can be passed
            as callable in which case it will be called by the default value to create the description at runtime.
            Defaults to "Scalars".
    """
    abs_diff = abs(actual - expected)
    rel_diff = float('inf') if expected == 0 else abs_diff / abs(expected)
    return _make_mismatch_msg(default_identifier='Scalars', identifier=identifier, extra=f'Expected {expected} but got {actual}.', abs_diff=abs_diff, atol=atol, rel_diff=rel_diff, rtol=rtol)