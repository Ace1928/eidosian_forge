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
@pytest.mark.slow
def test_2d_coord_names(self) -> None:
    self.plotmethod(x='x2d', y='y2d')
    ax = plt.gca()
    assert isinstance(ax, mpl_toolkits.mplot3d.axes3d.Axes3D)
    assert 'x2d' == ax.get_xlabel()
    assert 'y2d' == ax.get_ylabel()
    assert f'{self.darray.long_name} [{self.darray.units}]' == ax.get_zlabel()