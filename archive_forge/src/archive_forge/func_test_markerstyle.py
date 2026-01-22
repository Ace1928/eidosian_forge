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
def test_markerstyle(self):
    ax = self.df2.plot(marker='+')
    expected = _style_to_vertices('+')
    np.testing.assert_array_equal(expected, ax.collections[0].get_paths()[0].vertices)