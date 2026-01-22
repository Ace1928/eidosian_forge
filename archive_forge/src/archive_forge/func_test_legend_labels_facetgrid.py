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
def test_legend_labels_facetgrid(self) -> None:
    ds2 = self.ds.copy()
    ds2['hue'] = ['d', 'a', 'c', 'b']
    g = ds2.plot.scatter(x='A', y='B', hue='hue', markersize='x', col='col')
    legend = g.figlegend
    assert legend is not None
    actual = tuple((t.get_text() for t in legend.texts))
    expected = ('x [xunits]', '$\\mathdefault{0}$', '$\\mathdefault{1}$', '$\\mathdefault{2}$')
    assert actual == expected