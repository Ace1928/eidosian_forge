import datetime
import io
import os
import pathlib
import tempfile
from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
import pytz
from packaging.version import Version
from pandas.api.types import is_datetime64_any_dtype
from pandas.testing import assert_series_equal
from shapely.geometry import Point, Polygon, box
import geopandas
from geopandas import GeoDataFrame, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.io.file import _detect_driver, _EXTENSION_TO_DRIVER
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df
def test_read_file_bytesio(file_path, engine):
    file_binary_stream = open(file_path, 'rb')
    file_bytesio = io.BytesIO(open(file_path, 'rb').read())
    gdf_binary_stream = read_file(file_binary_stream, engine=engine)
    gdf_bytesio = read_file(file_bytesio, engine=engine)
    assert isinstance(gdf_binary_stream, geopandas.GeoDataFrame)
    assert isinstance(gdf_bytesio, geopandas.GeoDataFrame)