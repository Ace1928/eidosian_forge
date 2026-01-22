from pandas import DataFrame, Series
from shapely.geometry import Point
from geopandas import GeoDataFrame, GeoSeries
def test_geometry_multiple(self):
    assert type(self.df[['geometry', 'value1']]) is GeoDataFrame