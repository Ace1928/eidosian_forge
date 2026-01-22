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
def test_colorbar(self) -> None:
    vmin = self.darray.values.min()
    vmax = self.darray.values.max()
    expected = np.array((vmin, vmax))
    self.g.map_dataarray(xplt.imshow, 'x', 'y')
    for image in plt.gcf().findobj(mpl.image.AxesImage):
        assert isinstance(image, mpl.image.AxesImage)
        clim = np.array(image.get_clim())
        assert np.allclose(expected, clim)
    assert 1 == len(find_possible_colorbars())