from __future__ import annotations
import itertools
from typing import Any
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable
from xarray.core import indexing, nputils
from xarray.core.indexes import PandasIndex, PandasMultiIndex
from xarray.core.types import T_Xarray
from xarray.tests import (
def check_integer(indexer_cls):
    value = indexer_cls((1, np.uint64(2))).tuple
    assert all((isinstance(v, int) for v in value))
    assert value == (1, 2)