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
@pytest.mark.skipif(TEST_NEAREST, reason='This test can only be run _without_ PyGEOS >= 0.10 installed')
def test_no_nearest_all():
    df1 = geopandas.GeoDataFrame({'geometry': []})
    df2 = geopandas.GeoDataFrame({'geometry': []})
    with pytest.raises(NotImplementedError, match='Currently, only PyGEOS >= 0.10.0 or Shapely >= 2.0 supports'):
        sjoin_nearest(df1, df2)