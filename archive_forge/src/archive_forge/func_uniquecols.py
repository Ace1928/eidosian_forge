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
def uniquecols(data):
    """
    Return unique columns

    This is used for figuring out which columns are
    constant within a group
    """
    bool_idx = data.apply(lambda col: len(np.unique(col)) == 1, axis=0)
    data = data.loc[:, bool_idx].iloc[0:1, :].reset_index(drop=True)
    return data