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
def get_unique(self) -> list[T_PandasOrXarrayIndex]:
    """Return a list of unique indexes, preserving order."""
    unique_indexes: list[T_PandasOrXarrayIndex] = []
    seen: set[int] = set()
    for index in self._indexes.values():
        index_id = id(index)
        if index_id not in seen:
            unique_indexes.append(index)
            seen.add(index_id)
    return unique_indexes