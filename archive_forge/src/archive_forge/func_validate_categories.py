from __future__ import annotations
from datetime import (
from decimal import Decimal
import re
from typing import (
import warnings
import numpy as np
import pytz
from pandas._libs import (
from pandas._libs.interval import Interval
from pandas._libs.properties import cache_readonly
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.offsets import BDay
from pandas.compat import pa_version_under10p1
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.util import capitalize_first_letter
@staticmethod
def validate_categories(categories, fastpath: bool=False) -> Index:
    """
        Validates that we have good categories

        Parameters
        ----------
        categories : array-like
        fastpath : bool
            Whether to skip nan and uniqueness checks

        Returns
        -------
        categories : Index
        """
    from pandas.core.indexes.base import Index
    if not fastpath and (not is_list_like(categories)):
        raise TypeError(f"Parameter 'categories' must be list-like, was {repr(categories)}")
    if not isinstance(categories, ABCIndex):
        categories = Index._with_infer(categories, tupleize_cols=False)
    if not fastpath:
        if categories.hasnans:
            raise ValueError('Categorical categories cannot be null')
        if not categories.is_unique:
            raise ValueError('Categorical categories must be unique')
    if isinstance(categories, ABCCategoricalIndex):
        categories = categories.categories
    return categories