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
@pytest.mark.parametrize('kind', _pandas_kinds)
@check_figures_equal(extensions=['png', 'pdf'])
def test_pandas_kind(self, kind, fig_test, fig_ref):
    """Test Pandas kind."""
    import importlib
    _scipy_dependent_kinds = ['kde', 'density']
    _y_kinds = ['pie']
    _xy_kinds = ['scatter', 'hexbin']
    kwargs = {}
    if kind in _scipy_dependent_kinds:
        if not importlib.util.find_spec('scipy'):
            with pytest.raises(ModuleNotFoundError, match="No module named 'scipy'"):
                self.gdf.plot(kind=kind)
            return
    elif kind in _y_kinds:
        kwargs = {'y': 'y'}
    elif kind in _xy_kinds:
        kwargs = {'x': 'x', 'y': 'y'}
        if kind == 'hexbin':
            kwargs['gridsize'] = 10
    self.compare_figures(kind, fig_test, fig_ref, kwargs)
    plt.close('all')