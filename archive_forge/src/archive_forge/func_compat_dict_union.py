from __future__ import annotations
import contextlib
import functools
import inspect
import io
import itertools
import math
import os
import re
import sys
import warnings
from collections.abc import (
from enum import Enum
from pathlib import Path
from typing import (
import numpy as np
import pandas as pd
from xarray.namedarray.utils import (  # noqa: F401
def compat_dict_union(first_dict: Mapping[K, V], second_dict: Mapping[K, V], compat: Callable[[V, V], bool]=equivalent) -> MutableMapping[K, V]:
    """Return the union of two dictionaries as a new dictionary.

    An exception is raised if any keys are found in both dictionaries and the
    values are not compatible.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        Mappings to merge.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equivalence.

    Returns
    -------
    union : dict
        union of the contents.
    """
    new_dict = dict(first_dict)
    update_safety_check(first_dict, second_dict, compat)
    new_dict.update(second_dict)
    return new_dict