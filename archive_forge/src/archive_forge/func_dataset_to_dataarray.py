from __future__ import annotations
import collections
import itertools
import operator
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict
import numpy as np
from xarray.core.alignment import align
from xarray.core.coordinates import Coordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import Index
from xarray.core.merge import merge
from xarray.core.utils import is_dask_collection
from xarray.core.variable import Variable
def dataset_to_dataarray(obj: Dataset) -> DataArray:
    if not isinstance(obj, Dataset):
        raise TypeError(f'Expected Dataset, got {type(obj)}')
    if len(obj.data_vars) > 1:
        raise TypeError('Trying to convert Dataset with more than one data variable to DataArray')
    return next(iter(obj.data_vars.values()))