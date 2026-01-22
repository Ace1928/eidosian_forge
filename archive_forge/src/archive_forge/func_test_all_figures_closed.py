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
@pytest.fixture(scope='function', autouse=True)
def test_all_figures_closed():
    """meta-test to ensure all figures are closed at the end of a test

    Notes:  Scope is kept to module (only invoke this function once per test
    module) else tests cannot be run in parallel (locally). Disadvantage: only
    catches one open figure per run. May still give a false positive if tests
    are run in parallel.
    """
    yield None
    open_figs = len(plt.get_fignums())
    if open_figs:
        raise RuntimeError(f'tests did not close all figures ({open_figs} figures open)')