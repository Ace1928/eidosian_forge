import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_plot_with_projection_as_string_google_mercator(self):
    da = self.da.copy()
    plot = da.hvplot.image('x', 'y', crs=self.crs, projection='GOOGLE_MERCATOR')
    self.assert_projection(plot, 'merc')