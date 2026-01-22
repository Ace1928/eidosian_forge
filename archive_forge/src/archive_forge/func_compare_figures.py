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
def compare_figures(self, kind, fig_test, fig_ref, kwargs):
    """Compare Figures."""
    ax_pandas_1 = fig_test.subplots()
    self.df.plot(kind=kind, ax=ax_pandas_1, **kwargs)
    ax_geopandas_1 = fig_ref.subplots()
    self.gdf.plot(kind=kind, ax=ax_geopandas_1, **kwargs)
    ax_pandas_2 = fig_test.subplots()
    getattr(self.df.plot, kind)(ax=ax_pandas_2, **kwargs)
    ax_geopandas_2 = fig_ref.subplots()
    getattr(self.gdf.plot, kind)(ax=ax_geopandas_2, **kwargs)