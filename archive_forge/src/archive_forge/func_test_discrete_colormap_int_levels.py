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
def test_discrete_colormap_int_levels(self) -> None:
    for extend, levels, vmin, vmax, cmap in [('neither', 7, None, None, None), ('neither', 7, None, 20, mpl.colormaps['RdBu']), ('both', 7, 4, 8, None), ('min', 10, 4, 15, None)]:
        for kind in ['imshow', 'pcolormesh', 'contourf', 'contour']:
            primitive = getattr(self.darray.plot, kind)(levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
            assert levels >= len(primitive.norm.boundaries) - 1
            if vmax is None:
                assert primitive.norm.vmax >= self.data_max
            else:
                assert primitive.norm.vmax >= vmax
            if vmin is None:
                assert primitive.norm.vmin <= self.data_min
            else:
                assert primitive.norm.vmin <= vmin
            if kind != 'contour':
                assert extend == primitive.cmap.colorbar_extend
            else:
                assert 'max' == primitive.cmap.colorbar_extend
            assert levels >= len(primitive.cmap.colors)