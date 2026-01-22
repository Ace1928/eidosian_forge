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
def test_columns_no_geometry(tmpdir, file_format):
    """Reading a parquet file that is missing all of the geometry columns
    should raise a ValueError"""
    reader, writer = file_format
    test_dataset = 'naturalearth_lowres'
    df = read_file(get_path(test_dataset))
    filename = os.path.join(str(tmpdir), 'test.pq')
    writer(df, filename)
    with pytest.raises(ValueError):
        reader(filename, columns=['name'])