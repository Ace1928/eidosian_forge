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
def test_df_apply_geometry_dtypes(df):
    apply_types = []

    def get_dtypes(srs):
        apply_types.append((srs.name, type(srs)))
    df['geom2'] = df.geometry
    df.apply(get_dtypes)
    expected = [('geometry', GeoSeries), ('value1', pd.Series), ('value2', pd.Series), ('geom2', GeoSeries)]
    assert apply_types == expected