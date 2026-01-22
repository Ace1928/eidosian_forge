from contextlib import contextmanager
import glob
import os
import pathlib
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from geopandas import _compat as compat
import geopandas
from shapely.geometry import Point
@pytest.mark.skipif(not compat.HAS_PYGEOS, reason='requires pygeos to test #1745')
def test_pygeos_switch(tmpdir):
    with with_use_pygeos(False):
        gdf = _create_gdf()
        path = str(tmpdir / 'gdf_crs1.pickle')
        gdf.to_pickle(path)
        result = pd.read_pickle(path)
        assert_geodataframe_equal(result, gdf)
    with with_use_pygeos(False):
        gdf = _create_gdf()
        path = str(tmpdir / 'gdf_crs1.pickle')
        gdf.to_pickle(path)
    with with_use_pygeos(True):
        result = pd.read_pickle(path)
        gdf = _create_gdf()
        assert_geodataframe_equal(result, gdf)
    with with_use_pygeos(True):
        gdf = _create_gdf()
        path = str(tmpdir / 'gdf_crs1.pickle')
        gdf.to_pickle(path)
    with with_use_pygeos(False):
        result = pd.read_pickle(path)
        gdf = _create_gdf()
        assert_geodataframe_equal(result, gdf)