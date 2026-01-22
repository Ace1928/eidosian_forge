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
@pytest.mark.skip(reason='array-like style_kwds not supported for mixed geometry types (#1379)')
def test_style_kwargs_linewidth_listlike(self):
    for ax in [self.series.plot(linewidths=[2, 4, 5.5]), self.series.plot(linewidths=[2, 4, 5.5]), self.df.plot(linewidths=[2, 4, 5.5])]:
        np.testing.assert_array_equal([2, 4, 5.5], ax.collections[0].get_linewidths())