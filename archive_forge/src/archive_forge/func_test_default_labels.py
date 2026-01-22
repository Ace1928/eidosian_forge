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
def test_default_labels(self) -> None:
    g = self.ds.plot.scatter(x='A', y='B', row='row', col='col', hue='hue')
    for label, ax in zip(self.ds.coords['col'].values, g.axs[0, :]):
        assert substring_in_axes(str(label), ax)
    for ax in g.axs[-1, :]:
        assert ax.get_xlabel() == 'A [Aunits]'
    for ax in g.axs[:, 0]:
        assert ax.get_ylabel() == 'B [Bunits]'