import datetime as dt
from unittest import SkipTest, skipIf
import colorcet as cc
import numpy as np
import pandas as pd
import pytest
from numpy import nan
from packaging.version import Version
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import apply_when
from holoviews.streams import Tap
from holoviews.util import render
import logging
@pytest.mark.parametrize('agg_input_fn', (ds.first, ds.last, ds.min, ds.max))
def test_rasterize_where_agg_with_column(point_plot, agg_input_fn):
    agg_fn = ds.where(agg_input_fn('val'), 's')
    rast_input = dict(dynamic=False, x_range=(-1, 1), y_range=(-1, 1), width=2, height=2)
    img = rasterize(point_plot, aggregator=agg_fn, **rast_input)
    assert list(img.data) == ['s']
    img_no_column = rasterize(point_plot, aggregator=ds.where(agg_input_fn('val')), **rast_input)
    np.testing.assert_array_equal(img['s'], img_no_column['s'])