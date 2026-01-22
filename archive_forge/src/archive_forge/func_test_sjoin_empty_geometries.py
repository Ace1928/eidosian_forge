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
def test_sjoin_empty_geometries(self):
    empty = GeoDataFrame(geometry=[GeometryCollection()] * 3)
    df = sjoin(pd.concat([self.pointdf, empty]), self.polydf, how='left')
    assert df.shape == (24, 8)
    df2 = sjoin(self.pointdf, pd.concat([self.polydf, empty]), how='left')
    assert df2.shape == (21, 8)