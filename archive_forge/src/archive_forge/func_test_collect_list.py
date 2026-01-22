from shapely.geometry import LineString, MultiPoint, Point
from geopandas import GeoSeries
from geopandas.tools import collect
import pytest
def test_collect_list(self):
    result = collect([self.p1, self.p2, self.p3])
    assert self.mpc.equals(result)