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
def test_geom_equals_exact(self):
    assert np.all(self.g1.geom_equals_exact(self.g1, 0.001))
    assert_array_equal(self.g1.geom_equals_exact(self.sq, 0.001), [False, True])
    assert_array_equal(self.a1.geom_equals_exact(self.a2, 0.001, align=True), [False, True, False])
    assert_array_equal(self.a1.geom_equals_exact(self.a2, 0.001, align=False), [False, False])