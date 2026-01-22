from __future__ import annotations
import inspect
import itertools
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING, cast, overload
from warnings import warn
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping import aes
def from_inches(value: float, units: str) -> float:
    """
    Convert value in inches to given units

    Parameters
    ----------
    value : float
        Value to be converted
    units : str
        Units to convert value to. Must be one of
        `['in', 'cm', 'mm']`.
    """
    lookup: dict[str, Callable[[float], float]] = {'in': lambda x: x, 'cm': lambda x: x * 2.54, 'mm': lambda x: x * 2.54 * 10}
    try:
        return lookup[units](value)
    except KeyError as e:
        msg = f"Unknown units '{units}'"
        raise PlotnineError(msg) from e