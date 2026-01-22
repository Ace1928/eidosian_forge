import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_proj_to_cartopy(self):
    from ..util import proj_to_cartopy
    crs = proj_to_cartopy('+init=epsg:26911')
    assert isinstance(crs, self.ccrs.CRS)