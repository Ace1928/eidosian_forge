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
def test_datashade_rescale_discrete_levels(self):
    expected = False
    plot = self.df.hvplot(x='x', y='y', datashade=True, cnorm='eq_hist', rescale_discrete_levels=expected)
    actual = plot.callback.inputs[0].callback.operation.p['rescale_discrete_levels']
    assert actual is expected