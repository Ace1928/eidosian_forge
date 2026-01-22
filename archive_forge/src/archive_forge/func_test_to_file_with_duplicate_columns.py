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
def test_to_file_with_duplicate_columns(tmpdir, engine):
    df = GeoDataFrame(data=[[1, 2, 3]], columns=['a', 'b', 'a'], geometry=[Point(1, 1)])
    tempfilename = os.path.join(str(tmpdir), 'duplicate.shp')
    with pytest.raises(ValueError, match='GeoDataFrame cannot contain duplicated column names.'):
        df.to_file(tempfilename, engine=engine)