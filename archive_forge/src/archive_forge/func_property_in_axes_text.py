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
def property_in_axes_text(property, property_str, target_txt, ax):
    """
    Return True if the specified text in an axes
    has the property assigned to property_str
    """
    alltxt = ax.findobj(mpl.text.Text)
    check = []
    for t in alltxt:
        if t.get_text() == target_txt:
            check.append(plt.getp(t, property) == property_str)
    return all(check)