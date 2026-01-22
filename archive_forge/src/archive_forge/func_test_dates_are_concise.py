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
@pytest.mark.xfail(reason='Failing inside matplotlib. Should probably be fixed upstream because other plot functions can handle it. Remove this test when it works, already in Common2dMixin')
def test_dates_are_concise(self) -> None:
    import matplotlib.dates as mdates
    time = pd.date_range('2000-01-01', '2000-01-10')
    a = DataArray(np.random.randn(2, len(time)), [('xx', [1, 2]), ('t', time)])
    self.plotfunc(a, x='t')
    ax = plt.gca()
    assert isinstance(ax.xaxis.get_major_locator(), mdates.AutoDateLocator)
    assert isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)