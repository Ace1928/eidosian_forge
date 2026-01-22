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
def test_discrete_colormap_list_levels_and_vmin_or_vmax(self) -> None:
    levels = [0, 5, 10, 15]
    primitive = self.darray.plot(levels=levels, vmin=-3, vmax=20)
    assert primitive.norm.vmax == max(levels)
    assert primitive.norm.vmin == min(levels)