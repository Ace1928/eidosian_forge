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
def test_sjoin_predicate(self):
    df = sjoin(self.pointdf, self.polydf, how='left', predicate='within')
    assert df.shape == (21, 8)
    assert df.loc[1]['BoroName'] == 'Staten Island'
    df = sjoin(self.pointdf, self.polydf, how='left', predicate='contains')
    assert df.shape == (21, 8)
    assert np.isnan(df.loc[1]['Shape_Area'])