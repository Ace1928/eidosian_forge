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
def test__infer_interval_breaks_logscale_invalid_coords(self) -> None:
    """
        Check error is raised when passing non-positive coordinates with logscale
        """
    x = np.linspace(0, 5, 6)
    with pytest.raises(ValueError):
        _infer_interval_breaks(x, scale='log')
    x = np.linspace(-5, 5, 11)
    with pytest.raises(ValueError):
        _infer_interval_breaks(x, scale='log')