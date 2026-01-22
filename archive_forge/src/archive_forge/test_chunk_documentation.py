from __future__ import annotations
import pytest
import operator
import numpy as np
import dask.array as da
from dask.array.chunk import coarsen, getitem, keepdims_wrapper
https://github.com/dask/dask/issues/10274