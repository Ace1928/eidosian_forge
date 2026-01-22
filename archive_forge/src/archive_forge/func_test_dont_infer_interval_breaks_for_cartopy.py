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
def test_dont_infer_interval_breaks_for_cartopy(self) -> None:
    ax = plt.gca()
    setattr(ax, 'projection', True)
    artist = self.plotmethod(x='x2d', y='y2d', ax=ax)
    assert isinstance(artist, mpl.collections.QuadMesh)
    arr = artist.get_array()
    assert arr is not None
    assert arr.size <= self.darray.size