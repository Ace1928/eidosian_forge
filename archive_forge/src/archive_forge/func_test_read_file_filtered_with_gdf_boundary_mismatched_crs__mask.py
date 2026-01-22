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
def test_read_file_filtered_with_gdf_boundary_mismatched_crs__mask(df_nybb, engine):
    skip_pyogrio_not_supported(engine)
    full_df_shape = df_nybb.shape
    nybb_filename = geopandas.datasets.get_path('nybb')
    mask = geopandas.GeoDataFrame(geometry=[box(1031051.7879884212, 224272.49231459625, 1047224.3104931959, 244317.30894023244)], crs=NYBB_CRS)
    mask.to_crs(epsg=4326, inplace=True)
    filtered_df = read_file(nybb_filename, mask=mask.geometry, engine=engine)
    filtered_df_shape = filtered_df.shape
    assert full_df_shape != filtered_df_shape
    assert filtered_df_shape == (2, 5)