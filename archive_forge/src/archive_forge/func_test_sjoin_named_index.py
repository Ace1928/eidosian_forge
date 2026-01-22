import math
from typing import Sequence
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon, GeometryCollection
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, sjoin, sjoin_nearest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
@pytest.mark.parametrize('how', ['left', 'right', 'inner'])
def test_sjoin_named_index(self, how):
    pointdf2 = self.pointdf.copy()
    pointdf2.index.name = 'pointid'
    polydf = self.polydf.copy()
    polydf.index.name = 'polyid'
    res = sjoin(pointdf2, polydf, how=how)
    assert pointdf2.index.name == 'pointid'
    assert polydf.index.name == 'polyid'
    if how == 'right':
        assert res.index.name == 'polyid'
    else:
        assert res.index.name == 'pointid'