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
def test_estimate_utm_crs__projected(self):
    assert self.landmarks.to_crs('EPSG:3857').estimate_utm_crs() == CRS('EPSG:32618')