import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_plot_with_crs_as_attr_str(self):
    da = self.da.copy()
    da.rio._crs = False
    da.attrs = {'bar': self.crs}
    plot = da.hvplot.image('x', 'y', crs='bar')
    self.assertCRS(plot)