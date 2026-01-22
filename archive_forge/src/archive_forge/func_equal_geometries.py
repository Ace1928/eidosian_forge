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
def equal_geometries(result, expected):
    for r, e in zip(result, expected):
        if r is None or e is None:
            if not (r is None and e is None):
                return False
        elif not r.equals(e):
            return False
    return True