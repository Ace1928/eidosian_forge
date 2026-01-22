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
def test_negative_legend(self):
    ax = self.df.plot(column='NEGATIVES', scheme='FISHER_JENKS', k=3, cmap='OrRd', legend=True)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    expected = ['-10.00,  -3.41', ' -3.41,   3.30', '  3.30,  10.00']
    assert labels == expected