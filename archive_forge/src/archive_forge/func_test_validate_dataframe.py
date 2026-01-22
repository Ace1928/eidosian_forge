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
def test_validate_dataframe():
    test_dataset = 'naturalearth_lowres'
    df = read_file(get_path(test_dataset))
    _validate_dataframe(df)
    _validate_dataframe(df.set_index('iso_a3'))
    df[0] = 1
    with pytest.raises(ValueError):
        _validate_dataframe(df)
    with pytest.raises(ValueError):
        _validate_dataframe(df.set_index(0))
    with pytest.raises(ValueError):
        _validate_dataframe('not a dataframe')