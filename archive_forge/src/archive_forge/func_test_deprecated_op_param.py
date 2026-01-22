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
@pytest.mark.parametrize('dfs', ['default-index'], indirect=True)
@pytest.mark.parametrize('op', ['intersects', 'contains', 'within'])
def test_deprecated_op_param(self, dfs, op):
    _, df1, df2, _ = dfs
    with pytest.warns(FutureWarning, match='`op` parameter is deprecated'):
        sjoin(df1, df2, op=op)