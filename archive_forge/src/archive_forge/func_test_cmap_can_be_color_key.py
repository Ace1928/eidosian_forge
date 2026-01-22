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
def test_cmap_can_be_color_key(self):
    color_key = {'A': '#ff0000', 'B': '#00ff00', 'C': '#0000ff'}
    self.df.hvplot.points(x='x', y='y', by='category', cmap=color_key, datashade=True)
    with self.assertRaises(TypeError):
        self.df.hvplot.points(x='x', y='y', by='category', datashade=True, cmap='kbc_r', color_key=color_key)