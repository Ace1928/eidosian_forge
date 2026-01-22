import sys
from unittest import SkipTest
from parameterized import parameterized
import colorcet as cc
import holoviews as hv
import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import pytest
from holoviews import Store, render
from holoviews.element import Image, QuadMesh, Points
from holoviews.core.spaces import DynamicMap
from holoviews.core.overlay import Overlay
from holoviews.element.chart import Scatter
from holoviews.element.comparison import ComparisonTestCase
from hvplot.converter import HoloViewsConverter
from hvplot.tests.util import makeTimeDataFrame
from packaging.version import Version
def test_wide_charts_categorically_shaded_by(self):
    cat_col = 'category'
    plot = self.df.hvplot.scatter('x', 'y', by=cat_col, datashade=True)
    expected_cmap = HoloViewsConverter._default_cmaps['categorical']
    assert plot.callback.inputs[0].callback.operation.p.cmap == expected_cmap
    assert plot.callback.inputs[0].callback.operation.p.aggregator.column == cat_col