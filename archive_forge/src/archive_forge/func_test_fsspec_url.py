from __future__ import absolute_import
from itertools import product
import json
from packaging.version import Version
import os
import pathlib
import pytest
from pandas import DataFrame, read_parquet as pd_read_parquet
from pandas.testing import assert_frame_equal
import numpy as np
import pyproj
import shapely
from shapely.geometry import box, Point, MultiPolygon
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, read_file, read_parquet, read_feather
from geopandas.array import to_wkb
from geopandas.datasets import get_path
from geopandas.io.arrow import (
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import mock
def test_fsspec_url():
    fsspec = pytest.importorskip('fsspec')
    import fsspec.implementations.memory

    class MyMemoryFileSystem(fsspec.implementations.memory.MemoryFileSystem):

        def __init__(self, is_set, *args, **kwargs):
            self.is_set = is_set
            super().__init__(*args, **kwargs)
    fsspec.register_implementation('memory', MyMemoryFileSystem, clobber=True)
    memfs = MyMemoryFileSystem(is_set=True)
    test_dataset = 'naturalearth_lowres'
    df = read_file(get_path(test_dataset))
    with memfs.open('data.parquet', 'wb') as f:
        df.to_parquet(f)
    result = read_parquet('memory://data.parquet', storage_options={'is_set': True})
    assert_geodataframe_equal(result, df)
    result = read_parquet('memory://data.parquet', filesystem=memfs)
    assert_geodataframe_equal(result, df)
    fsspec.register_implementation('memory', fsspec.implementations.memory.MemoryFileSystem, clobber=True)