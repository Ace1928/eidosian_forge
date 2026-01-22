import os
from packaging.version import Version
import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import shapely
from shapely.geometry import Point, GeometryCollection, LineString, LinearRing
import geopandas
from geopandas import GeoDataFrame, GeoSeries
import geopandas._compat as compat
from geopandas.array import from_shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
@pytest.mark.skipif(shapely.geos.geos_version < (3, 9, 0), reason='requires GEOS>=3.9')
def test_repr_boxed_display_precision():
    p1 = Point(10.123456789, 50.123456789)
    p2 = Point(4.123456789, 20.123456789)
    s1 = GeoSeries([p1, p2, None])
    assert 'POINT (10.12346 50.12346)' in repr(s1)
    s3 = GeoSeries([p1, p2], crs=4326)
    assert 'POINT (10.12346 50.12346)' in repr(s3)
    p1 = Point(3000.123456789, 3000.123456789)
    p2 = Point(4000.123456789, 4000.123456789)
    s2 = GeoSeries([p1, p2, None])
    assert 'POINT (3000.123 3000.123)' in repr(s2)
    s4 = GeoSeries([p1, p2], crs=3857)
    assert 'POINT (3000.123 3000.123)' in repr(s4)
    geopandas.options.display_precision = 1
    assert 'POINT (10.1 50.1)' in repr(s1)
    geopandas.options.display_precision = 9
    assert 'POINT (10.123456789 50.123456789)' in repr(s1)