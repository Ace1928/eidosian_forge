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
def test_europp(self):
    europp = ccrs.EuroPP()
    proj4_init = europp.proj4_init
    assert '+proj=utm' in proj4_init
    assert '+zone=32' in proj4_init
    assert '+ellps=intl' in proj4_init