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
@pytest.mark.parametrize('time', datetime_type_tests, ids=('naive_datetime', 'datetime_with_timezone'))
@pytest.mark.parametrize('driver,ext', driver_ext_pairs)
def test_to_file_datetime(tmpdir, driver, ext, time, engine):
    """Test writing a data file with the datetime column type"""
    if engine == 'pyogrio' and time.tzinfo is not None:
        pytest.skip("pyogrio doesn't yet support timezones")
    if ext in ('.shp', ''):
        pytest.skip(f"Driver corresponding to ext {ext} doesn't support dt fields")
    tempfilename = os.path.join(str(tmpdir), f'test_datetime{ext}')
    point = Point(0, 0)
    df = GeoDataFrame({'a': [1.0, 2.0], 'b': [time, time]}, geometry=[point, point], crs=4326)
    fiona_precision_limit = 'ms'
    df['b'] = df['b'].dt.round(freq=fiona_precision_limit)
    df.to_file(tempfilename, driver=driver, engine=engine)
    df_read = read_file(tempfilename, engine=engine)
    assert_geodataframe_equal(df.drop(columns=['b']), df_read.drop(columns=['b']))
    if df['b'].dt.tz is not None:
        assert_series_equal(df['b'].dt.tz_convert(pytz.utc), df_read['b'].dt.tz_convert(pytz.utc))
    else:
        if engine == 'pyogrio' and PANDAS_GE_20:
            df['b'] = df['b'].astype('datetime64[ms]')
        assert_series_equal(df['b'], df_read['b'])