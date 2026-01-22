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
def test_when_datashade_is_true_set_hover_to_false_by_default(self):
    plot = self.df.hvplot(x='x', y='y', datashade=True)
    opts = Store.lookup_options('bokeh', plot[()], 'plot').kwargs
    assert 'hover' not in opts.get('tools')