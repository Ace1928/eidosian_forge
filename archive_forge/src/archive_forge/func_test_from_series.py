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
def test_from_series(self):
    shapes = [Polygon([(random.random(), random.random()) for _ in range(3)]) for _ in range(10)]
    with ignore_shapely2_warnings():
        s = pd.Series(shapes, index=list('abcdefghij'), name='foo')
    g = GeoSeries(s)
    check_geoseries(g)
    assert [a.equals(b) for a, b in zip(s, g)]
    assert s.name == g.name
    assert s.index is g.index