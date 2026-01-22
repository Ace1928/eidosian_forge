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
@pytest.mark.parametrize('ext', dt_exts)
def test_read_file_datetime_invalid(tmpdir, ext, engine):
    date_str = '9999-99-99T00:00:00'
    tempfilename = write_invalid_date_file(date_str, tmpdir, ext, engine)
    res = read_file(tempfilename)
    if ext == 'gpkg':
        assert is_datetime64_any_dtype(res['date'])
        assert pd.isna(res['date'].iloc[-1])
    else:
        assert res['date'].dtype == 'object'
        assert isinstance(res['date'].iloc[-1], str)