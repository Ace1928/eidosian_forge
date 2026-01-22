from shapely.geometry import LineString, MultiPoint, Point
from geopandas import GeoSeries
from geopandas.tools import collect
import pytest
def test_collect_single(self):
    result = collect(self.p1)
    assert self.p1.equals(result)