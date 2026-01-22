from __future__ import annotations
import contextlib
import inspect
import math
from collections.abc import Hashable
from copy import copy
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xarray.plot as xplt
from xarray import DataArray, Dataset
from xarray.namedarray.utils import module_available
from xarray.plot.dataarray_plot import _infer_interval_breaks
from xarray.plot.dataset_plot import _infer_meta_data
from xarray.plot.utils import (
from xarray.tests import (
def test_multiindex_raises_typeerror(self) -> None:
    a = DataArray(easy_array((3, 2)), dims=('x', 'y'), coords=dict(x=('x', [0, 1, 2]), a=('y', [0, 1]), b=('y', [2, 3])))
    a = a.set_index(y=('a', 'b'))
    with pytest.raises(TypeError, match='[Pp]lot'):
        self.plotfunc(a)