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
def test_project_point(self):
    point = sgeom.Point([0, 45])
    multi_point = sgeom.MultiPoint([point, sgeom.Point([180, 45])])
    pc = ccrs.PlateCarree()
    pc_rotated = ccrs.PlateCarree(central_longitude=180)
    result = pc_rotated.project_geometry(point, pc)
    assert_arr_almost_eq(result.xy, [[-180.0], [45.0]])
    result = pc_rotated.project_geometry(multi_point, pc)
    assert isinstance(result, sgeom.MultiPoint)
    assert len(result.geoms) == 2
    assert_arr_almost_eq(result.geoms[0].xy, [[-180.0], [45.0]])
    assert_arr_almost_eq(result.geoms[1].xy, [[0], [45.0]])