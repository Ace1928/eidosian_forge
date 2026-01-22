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
def test_positional_coord_string(self) -> None:
    self.plotmethod(y='x')
    ax = plt.gca()
    assert 'x_long_name [x_units]' == ax.get_ylabel()
    assert 'y_long_name [y_units]' == ax.get_xlabel()
    self.plotmethod(x='x')
    ax = plt.gca()
    assert 'x_long_name [x_units]' == ax.get_xlabel()
    assert 'y_long_name [y_units]' == ax.get_ylabel()