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
def test_2d_line_accepts_legend_kw(self) -> None:
    self.darray[:, :, 0].plot.line(x='dim_0', add_legend=False)
    assert not plt.gca().get_legend()
    plt.cla()
    self.darray[:, :, 0].plot.line(x='dim_0', add_legend=True)
    assert plt.gca().get_legend()
    assert plt.gca().get_legend().get_title().get_text() == 'dim_1'