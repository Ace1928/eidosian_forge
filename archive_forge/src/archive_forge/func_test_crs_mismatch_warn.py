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
def test_crs_mismatch_warn():
    t1 = T.copy()
    t2 = T.copy()
    t1.crs = 4326
    t2.crs = 3857
    with pytest.warns(UserWarning, match='CRS mismatch between the CRS'):
        _crs_mismatch_warn(t1, t2)
    with pytest.warns(UserWarning, match='CRS mismatch between the CRS'):
        _crs_mismatch_warn(T, t2)
    with pytest.warns(UserWarning, match='CRS mismatch between the CRS'):
        _crs_mismatch_warn(t1, T)