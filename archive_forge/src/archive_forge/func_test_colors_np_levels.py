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
def test_colors_np_levels(self) -> None:
    levels = np.array([-0.5, 0.0, 0.5, 1.0])
    artist = self.darray.plot.contour(levels=levels, colors=['k', 'r', 'w', 'b'])
    cmap = artist.cmap
    assert isinstance(cmap, mpl.colors.ListedColormap)
    colors = cmap.colors
    assert isinstance(colors, list)
    assert self._color_as_tuple(colors[1]) == (1.0, 0.0, 0.0)
    assert self._color_as_tuple(colors[2]) == (1.0, 1.0, 1.0)
    assert hasattr(cmap, '_rgba_over')
    assert self._color_as_tuple(cmap._rgba_over) == (0.0, 0.0, 1.0)