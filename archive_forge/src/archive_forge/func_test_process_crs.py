import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
@pytest.mark.parametrize('input', ['+init=epsg:26911'])
def test_process_crs(input):
    pytest.importorskip('pyproj')
    ccrs = pytest.importorskip('cartopy.crs')
    crs = process_crs(input)
    assert isinstance(crs, ccrs.CRS)