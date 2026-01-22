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
@requires_matplotlib
def test_plot1d_default_rcparams() -> None:
    import matplotlib as mpl
    ds = xr.tutorial.scatter_example_dataset(seed=42)
    with figure_context():
        fig, ax = plt.subplots(1, 1)
        ds.plot.scatter(x='A', y='B', marker='o', ax=ax)
        np.testing.assert_allclose(ax.collections[0].get_edgecolor(), mpl.colors.to_rgba_array('w'))
        fg = ds.plot.scatter(x='A', y='B', col='x', marker='o')
        ax = fg.axs.ravel()[0]
        np.testing.assert_allclose(ax.collections[0].get_edgecolor(), mpl.colors.to_rgba_array('w'))
        with assert_no_warnings():
            fig, ax = plt.subplots(1, 1)
            ds.plot.scatter(x='A', y='B', ax=ax, marker='x')
        fig, ax = plt.subplots(1, 1)
        ds.plot.scatter(x='A', y='B', marker='o', ax=ax, edgecolor='k')
        np.testing.assert_allclose(ax.collections[0].get_edgecolor(), mpl.colors.to_rgba_array('k'))