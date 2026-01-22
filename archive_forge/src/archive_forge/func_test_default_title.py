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
def test_default_title(self) -> None:
    a = DataArray(easy_array((4, 3, 2)), dims=['a', 'b', 'c'])
    a.coords['c'] = [0, 1]
    a.coords['d'] = 'foo'
    self.plotfunc(a.isel(c=1))
    title = plt.gca().get_title()
    assert 'c = 1, d = foo' == title or 'd = foo, c = 1' == title