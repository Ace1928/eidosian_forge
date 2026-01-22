import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_geo_hexbin(self):
    hextiles = self.df.hvplot.hexbin('x', 'y', geo=True)
    self.assertEqual(hextiles.crs, self.crs)