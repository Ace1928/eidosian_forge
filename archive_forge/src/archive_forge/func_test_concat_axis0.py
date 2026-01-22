import warnings
import pandas as pd
import pyproj
import pytest
from geopandas._compat import PANDAS_GE_21
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_index_equal
from shapely.geometry import Point
from geopandas import GeoDataFrame, GeoSeries
def test_concat_axis0(self):
    res = pd.concat([self.gdf, self.gdf])
    assert res.shape == (6, 2)
    assert isinstance(res, GeoDataFrame)
    assert isinstance(res.geometry, GeoSeries)
    self._check_metadata(res)
    exp = GeoDataFrame(pd.concat([pd.DataFrame(self.gdf), pd.DataFrame(self.gdf)]))
    assert_geodataframe_equal(exp, res)
    res = pd.concat([self.gdf.geometry, self.gdf.geometry])
    assert res.shape == (6,)
    assert isinstance(res, GeoSeries)
    assert isinstance(res.geometry, GeoSeries)