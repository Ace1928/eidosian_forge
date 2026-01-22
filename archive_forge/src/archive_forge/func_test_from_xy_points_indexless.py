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
def test_from_xy_points_indexless(self):
    x = np.array([0.0, 3.0])
    y = np.array([2.0, 5.0])
    z = np.array([-1.0, 4.0])
    expected = GeoSeries([Point(0, 2, -1), Point(3, 5, 4)])
    assert_geoseries_equal(expected, GeoSeries.from_xy(x, y, z))