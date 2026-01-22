from __future__ import annotations
import datetime
import typing
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.types import SideOptions
def exact_cftime_datetime_difference(a: CFTimeDatetime, b: CFTimeDatetime):
    """Exact computation of b - a

    Assumes:

        a = a_0 + a_m
        b = b_0 + b_m

    Here a_0, and b_0 represent the input dates rounded
    down to the nearest second, and a_m, and b_m represent
    the remaining microseconds associated with date a and
    date b.

    We can then express the value of b - a as:

        b - a = (b_0 + b_m) - (a_0 + a_m) = b_0 - a_0 + b_m - a_m

    By construction, we know that b_0 - a_0 must be a round number
    of seconds.  Therefore we can take the result of b_0 - a_0 using
    ordinary cftime.datetime arithmetic and round to the nearest
    second.  b_m - a_m is the remainder, in microseconds, and we
    can simply add this to the rounded timedelta.

    Parameters
    ----------
    a : cftime.datetime
        Input datetime
    b : cftime.datetime
        Input datetime

    Returns
    -------
    datetime.timedelta
    """
    seconds = b.replace(microsecond=0) - a.replace(microsecond=0)
    seconds = int(round(seconds.total_seconds()))
    microseconds = b.microsecond - a.microsecond
    return datetime.timedelta(seconds=seconds, microseconds=microseconds)