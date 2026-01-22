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
@pytest.mark.parametrize('driver,ext', driver_ext_pairs)
def test_to_file_bool(tmpdir, driver, ext, engine):
    """Test error raise when writing with a boolean column (GH #437)."""
    tempfilename = os.path.join(str(tmpdir), 'temp.{0}'.format(ext))
    df = GeoDataFrame({'col': [True, False, True], 'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)]}, crs=4326)
    df.to_file(tempfilename, driver=driver, engine=engine)
    result = read_file(tempfilename, engine=engine)
    if ext in ('.shp', ''):
        if engine == 'fiona':
            df['col'] = df['col'].astype('int64')
        else:
            df['col'] = df['col'].astype('int32')
    assert_geodataframe_equal(result, df)
    assert_correct_driver(tempfilename, ext, engine)