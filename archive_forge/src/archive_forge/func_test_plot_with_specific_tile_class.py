import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_plot_with_specific_tile_class(self):
    plot = self.df.hvplot.points('x', 'y', geo=False, tiles=hv.element.tiles.EsriImagery)
    self.assertEqual(len(plot), 2)
    self.assertIsInstance(plot.get(0), hv.Tiles)
    self.assertIn('ArcGIS', plot.get(0).data)