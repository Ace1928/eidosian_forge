from __future__ import annotations
import collections.abc
import copy
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
import numpy as np
import pandas as pd
from xarray.core import formatting, nputils, utils
from xarray.core.indexing import (
from xarray.core.utils import (
def to_pandas_indexes(self) -> Indexes[pd.Index]:
    """Returns an immutable proxy for Dataset or DataArrary pandas indexes.

        Raises an error if this proxy contains indexes that cannot be coerced to
        pandas.Index objects.

        """
    indexes: dict[Hashable, pd.Index] = {}
    for k, idx in self._indexes.items():
        if isinstance(idx, pd.Index):
            indexes[k] = idx
        elif isinstance(idx, Index):
            indexes[k] = idx.to_pandas_index()
    return Indexes(indexes, self._variables, index_type=pd.Index)