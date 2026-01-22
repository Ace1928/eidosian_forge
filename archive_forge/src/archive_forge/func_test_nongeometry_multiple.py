from pandas import DataFrame, Series
from shapely.geometry import Point
from geopandas import GeoDataFrame, GeoSeries
def test_nongeometry_multiple(self):
    assert type(self.df[['value1', 'value2']]) is DataFrame