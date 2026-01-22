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
def test_transform_points_xyz(self):
    rx = np.array([2574325.16])
    ry = np.array([837562.0])
    rz = np.array([5761325.0])
    src_proj = ccrs.Geocentric()
    target_proj = ccrs.Geodetic()
    res = target_proj.transform_points(x=rx, y=ry, z=rz, src_crs=src_proj)
    glat = res[..., 0]
    glon = res[..., 1]
    galt = res[..., 2]
    solx = np.array([18.0224043189])
    soly = np.array([64.9796515089])
    solz = np.array([5048.03893734])
    assert_arr_almost_eq(glat, solx)
    assert_arr_almost_eq(glon, soly)
    assert_arr_almost_eq(galt, solz)