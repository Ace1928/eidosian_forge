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
def test_2d_line_accepts_hue_kw(self) -> None:
    self.darray[:, :, 0].plot.line(hue='dim_0')
    assert plt.gca().get_legend().get_title().get_text() == 'dim_0'
    plt.cla()
    self.darray[:, :, 0].plot.line(hue='dim_1')
    assert plt.gca().get_legend().get_title().get_text() == 'dim_1'