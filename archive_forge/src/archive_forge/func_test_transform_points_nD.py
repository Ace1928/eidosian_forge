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
def test_transform_points_nD(self):
    rlons = np.array([[350.0, 352.0, 354.0], [350.0, 352.0, 354.0]])
    rlats = np.array([[-5.0, -0.0, 1.0], [-4.0, -1.0, 0.0]])
    src_proj = ccrs.RotatedGeodetic(pole_longitude=178.0, pole_latitude=38.0)
    target_proj = ccrs.Geodetic()
    res = target_proj.transform_points(x=rlons, y=rlats, src_crs=src_proj)
    unrotated_lon = res[..., 0]
    unrotated_lat = res[..., 1]
    solx = np.array([[-16.42176094, -14.85892262, -11.9062752], [-16.71055023, -14.58434624, -11.68799988]])
    soly = np.array([[46.00724251, 51.29188893, 52.59101488], [46.98728486, 50.30706042, 51.60004528]])
    assert_arr_almost_eq(unrotated_lon, solx)
    assert_arr_almost_eq(unrotated_lat, soly)