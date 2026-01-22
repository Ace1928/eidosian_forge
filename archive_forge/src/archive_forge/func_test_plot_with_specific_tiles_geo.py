import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_plot_with_specific_tiles_geo(self):
    import geoviews as gv
    plot = self.df.hvplot.points('x', 'y', geo=True, tiles='ESRI')
    self.assertEqual(len(plot), 2)
    self.assertIsInstance(plot.get(0), gv.element.WMTS)
    self.assertIn('ArcGIS', plot.get(0).data)