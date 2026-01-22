from __future__ import annotations
import os
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING, ClassVar, Literal, overload
import numpy as np
from monty.design_patterns import cached_class
from monty.serialization import loadfn
from pymatgen.util.string import Stringify
def in_array_list(array_list: list[np.ndarray] | np.ndarray, arr: np.ndarray, tol: float=1e-05) -> bool:
    """Extremely efficient nd-array comparison using numpy's broadcasting. This
    function checks if a particular array a, is present in a list of arrays.
    It works for arrays of any size, e.g., even matrix searches.

    Args:
        array_list ([array]): A list of arrays to compare to.
        arr (array): The test array for comparison.
        tol (float): The tolerance. Defaults to 1e-5. If 0, an exact match is done.

    Returns:
        bool: True if arr is in array_list.
    """
    if len(array_list) == 0:
        return False
    axes = tuple(range(1, arr.ndim + 1))
    if not tol:
        return any(np.all(array_list == arr[None, :], axes))
    return any(np.sum(np.abs(array_list - arr[None, :]), axes) < tol)