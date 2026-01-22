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
def test_repr_all_missing():
    s = GeoSeries([None, None, None])
    assert 'None' in repr(s)
    df = GeoDataFrame({'a': [1, 2, 3], 'geometry': s})
    assert 'None' in repr(df)
    assert 'geometry' in df._repr_html_()