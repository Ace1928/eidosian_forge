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
def test_markersize(self):
    ax = self.points.plot(markersize=10)
    assert ax.collections[0].get_sizes() == [10]
    ax = self.df.plot(markersize=10)
    assert ax.collections[0].get_sizes() == [10]
    ax = self.df.plot(column='values', markersize=10)
    assert ax.collections[0].get_sizes() == [10]
    ax = self.df.plot(markersize='values')
    assert (ax.collections[0].get_sizes() == self.df['values']).all()
    ax = self.df.plot(column='values', markersize='values')
    assert (ax.collections[0].get_sizes() == self.df['values']).all()