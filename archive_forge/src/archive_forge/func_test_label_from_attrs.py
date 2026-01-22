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
def test_label_from_attrs(self) -> None:
    da = self.darray.copy()
    assert '' == label_from_attrs(da)
    da.name = 0
    assert '0' == label_from_attrs(da)
    da.name = 'a'
    da.attrs['units'] = 'a_units'
    da.attrs['long_name'] = 'a_long_name'
    da.attrs['standard_name'] = 'a_standard_name'
    assert 'a_long_name [a_units]' == label_from_attrs(da)
    da.attrs.pop('long_name')
    assert 'a_standard_name [a_units]' == label_from_attrs(da)
    da.attrs.pop('units')
    assert 'a_standard_name' == label_from_attrs(da)
    da.attrs['units'] = 'a_units'
    da.attrs.pop('standard_name')
    assert 'a [a_units]' == label_from_attrs(da)
    da.attrs.pop('units')
    assert 'a' == label_from_attrs(da)
    long_latex_name = '$Ra_s = \\mathrm{mean}(\\epsilon_k) / \\mu M^2_\\infty$'
    da.attrs = dict(long_name=long_latex_name)
    assert label_from_attrs(da) == long_latex_name