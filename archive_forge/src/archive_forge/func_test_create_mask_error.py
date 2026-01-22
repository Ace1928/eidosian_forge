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
def test_create_mask_error() -> None:
    with pytest.raises(TypeError, match='unexpected key type'):
        indexing.create_mask((1, 2), (3, 4))