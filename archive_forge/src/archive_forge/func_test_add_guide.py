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
@pytest.mark.parametrize('add_guide, hue_style, legend, colorbar', [(None, None, False, True), (False, None, False, False), (True, None, False, True), (True, 'continuous', False, True), (False, 'discrete', False, False), (True, 'discrete', True, False)])
def test_add_guide(self, add_guide: bool | None, hue_style: Literal['continuous', 'discrete', None], legend: bool, colorbar: bool) -> None:
    meta_data = _infer_meta_data(self.ds, x='A', y='B', hue='hue', hue_style=hue_style, add_guide=add_guide, funcname='scatter')
    assert meta_data['add_legend'] is legend
    assert meta_data['add_colorbar'] is colorbar