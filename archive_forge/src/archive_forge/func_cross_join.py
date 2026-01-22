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
def cross_join(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cross between df1 & df2 if each is not empty
    """
    if len(df1) == 0:
        return df2
    if len(df2) == 0:
        return df1
    return df1.join(df2, how='cross')