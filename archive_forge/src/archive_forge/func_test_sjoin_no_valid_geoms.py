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
@pytest.mark.parametrize('predicate', ['intersects', 'within', 'contains'])
def test_sjoin_no_valid_geoms(self, predicate):
    """Tests a completely empty GeoDataFrame."""
    empty = GeoDataFrame(geometry=[], crs=self.pointdf.crs)
    assert sjoin(self.pointdf, empty, how='inner', predicate=predicate).empty
    assert sjoin(self.pointdf, empty, how='right', predicate=predicate).empty
    assert sjoin(empty, self.pointdf, how='inner', predicate=predicate).empty
    assert sjoin(empty, self.pointdf, how='left', predicate=predicate).empty