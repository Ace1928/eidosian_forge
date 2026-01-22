import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_check_crs():
    pytest.importorskip('pyproj')
    p = check_crs('epsg:26915 +units=m')
    assert p.srs == '+proj=utm +zone=15 +datum=NAD83 +units=m +no_defs'
    p = check_crs('wrong')
    assert p is None