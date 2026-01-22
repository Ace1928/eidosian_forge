from __future__ import annotations
import itertools
import typing
from contextlib import suppress
from typing import List
from warnings import warn
import numpy as np
import pandas.api.types as pdtypes
from .._utils import array_kind
from .._utils.registry import Registry
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import aes_to_scale
from .scale import scale
def scale_type(series):
    """
    Get a suitable scale for the series
    """
    if array_kind.continuous(series):
        stype = 'continuous'
    elif array_kind.ordinal(series):
        stype = 'ordinal'
    elif array_kind.discrete(series):
        stype = 'discrete'
    elif array_kind.datetime(series):
        stype = 'datetime'
    elif array_kind.timedelta(series):
        stype = 'timedelta'
    else:
        msg = "Don't know how to automatically pick scale for object of type {}. Defaulting to 'continuous'"
        warn(msg.format(series.dtype), PlotnineWarning)
        stype = 'continuous'
    return stype