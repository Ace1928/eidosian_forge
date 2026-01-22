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
def test_estimate_utm_crs__missing_crs(self):
    with pytest.raises(RuntimeError, match='crs must be set'):
        from_shapely([shapely.geometry.Polygon([(0, 90), (1, 90), (2, 90)])]).estimate_utm_crs()