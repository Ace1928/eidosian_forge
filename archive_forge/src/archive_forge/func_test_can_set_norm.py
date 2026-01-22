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
@pytest.mark.filterwarnings('ignore')
def test_can_set_norm(self) -> None:
    norm = mpl.colors.SymLogNorm(0.1)
    self.g.map_dataarray(xplt.imshow, 'x', 'y', norm=norm)
    for image in plt.gcf().findobj(mpl.image.AxesImage):
        assert isinstance(image, mpl.image.AxesImage)
        assert image.norm is norm