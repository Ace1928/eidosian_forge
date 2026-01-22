import copy
from io import BytesIO
import os
from pathlib import Path
import pickle
import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from numpy.testing import assert_array_almost_equal as assert_arr_almost_eq
import pyproj
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
def test_transform_points_empty():
    """Test CRS.transform_points with empty array."""
    crs = ccrs.Stereographic()
    result = crs.transform_points(ccrs.PlateCarree(), np.array([]), np.array([]))
    assert_array_equal(result, np.array([], dtype=np.float64).reshape(0, 3))