from __future__ import annotations
import itertools
import warnings
import pytest
from tlz import merge
import dask
import dask.array as da
from dask import config
from dask.array.chunk import getitem
from dask.array.slicing import (
from dask.array.utils import assert_eq, same_keys
Reproducer for https://github.com/dask/dask/issues/3730.

    Mutating based on an array with different chunks can cause new chunks to be
    used.  We need to ensure those new chunk sizes are applied to the mutated
    array, otherwise the array won't generate the correct keys.
    