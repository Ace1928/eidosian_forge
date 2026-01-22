import warnings
from typing import Any
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.groupby.base import transformation_kernels
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default import DefaultMethod
@classmethod
def is_aggregate(cls, key):
    """Check whether `key` is an alias for pandas.GroupBy.aggregation method."""
    return key in cls.agg_aliases