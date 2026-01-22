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
def order_as_data_mapping(arg1: DataLike | aes | None, arg2: DataLike | aes | None) -> tuple[DataLike | None, aes | None]:
    """
    Reorder args to ensure (data, mapping) order

    This function allow the user to pass mapping and data
    to ggplot and geom in any order.

    Parameter
    ---------
    arg1 : pd.DataFrame | aes
        Dataframe or aes Mapping
    arg2 : pd.DataFrame | aes
        Dataframe or aes Mapping

    Returns
    -------
    data : pd.DataFrame | callable
    mapping : aes
    """
    data: DataLike | None = None
    mapping: aes | None = None
    for arg in [arg1, arg2]:
        if isinstance(arg, aes):
            if mapping is None:
                mapping = arg
            else:
                raise TypeError('Expected a single aesthetic mapping, found two')
        elif is_data_like(arg):
            if data is None:
                data = arg
            else:
                raise TypeError('Expected a single dataframe, found two')
        elif arg is not None:
            raise TypeError(f'Bad type of argument {arg!r}, expected a dataframe or a mapping.')
    return (data, mapping)