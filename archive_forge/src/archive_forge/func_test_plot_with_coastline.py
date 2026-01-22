import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_plot_with_coastline(self):
    import geoviews as gv
    plot = self.df.hvplot.points('x', 'y', geo=True, coastline=True)
    self.assertEqual(len(plot), 2)
    coastline = plot.get(1)
    self.assertIsInstance(coastline, gv.Feature)