import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_plot_with_tiles(self):
    plot = self.df.hvplot.points('x', 'y', geo=False, tiles=True)
    self.assertEqual(len(plot), 2)
    self.assertIsInstance(plot.get(0), hv.Tiles)
    self.assertIn('openstreetmap', plot.get(0).data)