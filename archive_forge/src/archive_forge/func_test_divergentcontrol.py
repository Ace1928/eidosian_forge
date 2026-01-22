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
def test_divergentcontrol(self) -> None:
    neg = self.data - 0.1
    pos = self.data
    cmap_params = _determine_cmap_params(pos)
    assert cmap_params['vmin'] == 0
    assert cmap_params['vmax'] == 1
    assert cmap_params['cmap'] == 'viridis'
    cmap_params = _determine_cmap_params(neg)
    assert cmap_params['vmin'] == -0.9
    assert cmap_params['vmax'] == 0.9
    assert cmap_params['cmap'] == 'RdBu_r'
    cmap_params = _determine_cmap_params(neg, vmin=-0.1, center=False)
    assert cmap_params['vmin'] == -0.1
    assert cmap_params['vmax'] == 0.9
    assert cmap_params['cmap'] == 'viridis'
    cmap_params = _determine_cmap_params(neg, vmax=0.5, center=False)
    assert cmap_params['vmin'] == -0.1
    assert cmap_params['vmax'] == 0.5
    assert cmap_params['cmap'] == 'viridis'
    cmap_params = _determine_cmap_params(neg, center=False)
    assert cmap_params['vmin'] == -0.1
    assert cmap_params['vmax'] == 0.9
    assert cmap_params['cmap'] == 'viridis'
    cmap_params = _determine_cmap_params(neg, center=0)
    assert cmap_params['vmin'] == -0.9
    assert cmap_params['vmax'] == 0.9
    assert cmap_params['cmap'] == 'RdBu_r'
    cmap_params = _determine_cmap_params(neg, vmin=-0.1)
    assert cmap_params['vmin'] == -0.1
    assert cmap_params['vmax'] == 0.1
    assert cmap_params['cmap'] == 'RdBu_r'
    cmap_params = _determine_cmap_params(neg, vmax=0.5)
    assert cmap_params['vmin'] == -0.5
    assert cmap_params['vmax'] == 0.5
    assert cmap_params['cmap'] == 'RdBu_r'
    cmap_params = _determine_cmap_params(neg, vmax=0.6, center=0.1)
    assert cmap_params['vmin'] == -0.4
    assert cmap_params['vmax'] == 0.6
    assert cmap_params['cmap'] == 'RdBu_r'
    cmap_params = _determine_cmap_params(pos, vmin=-0.1)
    assert cmap_params['vmin'] == -0.1
    assert cmap_params['vmax'] == 0.1
    assert cmap_params['cmap'] == 'RdBu_r'
    cmap_params = _determine_cmap_params(pos, vmin=0.1)
    assert cmap_params['vmin'] == 0.1
    assert cmap_params['vmax'] == 1
    assert cmap_params['cmap'] == 'viridis'
    cmap_params = _determine_cmap_params(pos, vmax=0.5)
    assert cmap_params['vmin'] == 0
    assert cmap_params['vmax'] == 0.5
    assert cmap_params['cmap'] == 'viridis'
    cmap_params = _determine_cmap_params(neg, vmin=-0.2, vmax=0.6)
    assert cmap_params['vmin'] == -0.2
    assert cmap_params['vmax'] == 0.6
    assert cmap_params['cmap'] == 'viridis'
    cmap_params = _determine_cmap_params(pos, levels=[-0.1, 0, 1])
    assert cmap_params['cmap'].name == 'RdBu_r'