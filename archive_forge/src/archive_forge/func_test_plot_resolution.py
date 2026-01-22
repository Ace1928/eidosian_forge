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
@parameterized.expand([('image', Image), ('quadmesh', QuadMesh)])
def test_plot_resolution(self, kind, element):
    plot = self.da.hvplot(kind=kind)
    assert all(plot.data.x.diff('x').round(0) == 1)
    assert all(plot.data.y.diff('y').round(0) == 1)