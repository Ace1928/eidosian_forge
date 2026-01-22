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
def test_empty_join(self):
    polygons = geopandas.GeoDataFrame({'col2': [1, 2], 'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])]})
    not_in = geopandas.GeoDataFrame({'col1': [1], 'geometry': [Point(-0.5, 0.5)]})
    empty = sjoin(not_in, polygons, how='left', predicate='intersects')
    assert empty.index_right.isnull().all()
    empty = sjoin(not_in, polygons, how='right', predicate='intersects')
    assert empty.index_left.isnull().all()
    empty = sjoin(not_in, polygons, how='inner', predicate='intersects')
    assert empty.empty