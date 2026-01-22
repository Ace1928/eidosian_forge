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
def test_estimate_utm_crs__antimeridian(self):
    antimeridian = from_shapely([shapely.geometry.Point(1722483.900174921, 5228058.6143420935), shapely.geometry.Point(4624385.494808555, 8692574.544944234)], crs='EPSG:3851')
    assert antimeridian.estimate_utm_crs() == CRS('EPSG:32760')