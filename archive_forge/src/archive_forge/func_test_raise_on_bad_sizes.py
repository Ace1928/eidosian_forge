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
def test_raise_on_bad_sizes():
    with pytest.raises(ValueError) as info:
        T.contains(P)
    assert 'lengths' in str(info.value).lower()
    assert '12' in str(info.value)
    assert '21' in str(info.value)