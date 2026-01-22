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
def keep_levels(self, level_variables: Mapping[Any, Variable]) -> PandasMultiIndex | PandasIndex:
    """Keep only the provided levels and return a new multi-index with its
        corresponding coordinates.

        """
    index = self.index.droplevel([k for k in self.index.names if k not in level_variables])
    if isinstance(index, pd.MultiIndex):
        level_coords_dtype = {k: self.level_coords_dtype[k] for k in index.names}
        return self._replace(index, level_coords_dtype=level_coords_dtype)
    else:
        return PandasIndex(index.rename(self.dim), self.dim, coord_dtype=self.level_coords_dtype[index.name])