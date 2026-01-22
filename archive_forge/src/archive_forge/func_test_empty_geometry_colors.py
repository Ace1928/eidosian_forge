import itertools
from packaging.version import Version
import warnings
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.affinity import rotate
from shapely.geometry import (
from geopandas import GeoDataFrame, GeoSeries, read_file
from geopandas.datasets import get_path
import geopandas._compat as compat
from geopandas.plotting import GeoplotAccessor
import pytest
import matplotlib.pyplot as plt
@pytest.mark.parametrize('geoms', [[box(0, 0, 1, 1), box(7, 7, 8, 8)], [LineString([(1, 1), (1, 2)]), LineString([(7, 1), (7, 2)])], [Point(1, 1), Point(7, 7)]])
def test_empty_geometry_colors(self, geoms):
    s = GeoSeries(geoms, index=['r', 'b'])
    s2 = s.intersection(box(5, 0, 10, 10))
    ax = s2.plot(color=['red', 'blue'])
    blue = np.array([0.0, 0.0, 1.0, 1.0])
    if s.geom_type['r'] == 'LineString':
        np.testing.assert_array_equal(ax.get_children()[0].get_edgecolor()[0], blue)
    else:
        np.testing.assert_array_equal(ax.get_children()[0].get_facecolor()[0], blue)