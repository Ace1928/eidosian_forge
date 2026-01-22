import random
import numpy as np
import pandas as pd
from pyproj import CRS
import shapely
import shapely.affinity
import shapely.geometry
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE, BaseGeometry
import shapely.wkb
import shapely.wkt
import geopandas
from geopandas.array import (
import geopandas._compat as compat
import pytest
def test_as_array():
    arr = from_shapely(points_no_missing)
    np_arr1 = np.asarray(arr)
    np_arr2 = arr.to_numpy()
    assert np_arr1[0] == arr[0]
    np.testing.assert_array_equal(np_arr1, np_arr2)