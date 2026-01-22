import json
import os
import random
import re
import shutil
import tempfile
import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.testing import assert_index_equal
from pyproj import CRS
from shapely.geometry import (
from shapely.geometry.base import BaseGeometry
from geopandas import GeoSeries, GeoDataFrame, read_file, datasets, clip
from geopandas._compat import ignore_shapely2_warnings
from geopandas.array import GeometryArray, GeometryDtype
from geopandas.testing import assert_geoseries_equal, geom_almost_equals
from geopandas.tests.util import geom_equals
from pandas.testing import assert_series_equal
import pytest
def test_from_xy_points_w_z(self):
    index_values = [5, 6, 7]
    x = pd.Series([0, -1, 2], index=index_values)
    y = pd.Series([8, 3, 1], index=index_values)
    z = pd.Series([5, -6, 7], index=index_values)
    expected = GeoSeries([Point(0, 8, 5), Point(-1, 3, -6), Point(2, 1, 7)], index=index_values)
    assert_geoseries_equal(expected, GeoSeries.from_xy(x, y, z))