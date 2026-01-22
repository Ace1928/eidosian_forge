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
def test_equally_formatted_bins(self):
    ax = self.nybb.plot('vals', scheme='quantiles', legend=True)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    expected = ['0.00, 0.00', '0.00, 0.00', '0.00, 0.00', '0.00, 0.00', '0.00, 0.01']
    assert labels == expected
    ax2 = self.nybb.plot('vals', scheme='quantiles', legend=True, legend_kwds={'fmt': '{:.3f}'})
    labels = [t.get_text() for t in ax2.get_legend().get_texts()]
    expected = ['0.001, 0.002', '0.002, 0.003', '0.003, 0.003', '0.003, 0.004', '0.004, 0.005']
    assert labels == expected