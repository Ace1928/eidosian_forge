from __future__ import annotations
from collections import defaultdict
from numbers import Integral
import pandas as pd
from pandas.api.types import is_scalar
from tlz import partition_all
from dask.base import compute_as_if_collection, tokenize
from dask.dataframe import methods
from dask.dataframe.accessor import Accessor
from dask.dataframe.dispatch import (  # noqa: F401
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
@property
def known(self):
    """Whether the categories are fully known"""
    return has_known_categories(self._series)