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
def test_read_file_filtered__bbox(df_nybb, engine):
    nybb_filename = geopandas.datasets.get_path('nybb')
    bbox = (1031051.7879884212, 224272.49231459625, 1047224.3104931959, 244317.30894023244)
    filtered_df = read_file(nybb_filename, bbox=bbox, engine=engine)
    expected = df_nybb[df_nybb['BoroName'].isin(['Bronx', 'Queens'])]
    assert_geodataframe_equal(filtered_df, expected.reset_index(drop=True))