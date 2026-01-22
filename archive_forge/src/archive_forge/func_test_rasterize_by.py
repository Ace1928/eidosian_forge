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
def test_rasterize_by(self):
    if Version(hv.__version__) < Version('1.18.0a1'):
        raise SkipTest('hv.ImageStack introduced after 1.18.0a1')
    from holoviews.element import ImageStack
    expected = 'category'
    plot = self.df.hvplot(x='x', y='y', by=expected, rasterize=True, dynamic=False)
    assert isinstance(plot, ImageStack)
    assert plot.opts['cmap'] == cc.palette['glasbey_category10']