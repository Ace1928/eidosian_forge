import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_instantiate_crs_google_mercator():
    ccrs = pytest.importorskip('cartopy.crs')
    assert instantiate_crs_str('GOOGLE_MERCATOR') == ccrs.GOOGLE_MERCATOR
    assert instantiate_crs_str('google_mercator') == ccrs.GOOGLE_MERCATOR