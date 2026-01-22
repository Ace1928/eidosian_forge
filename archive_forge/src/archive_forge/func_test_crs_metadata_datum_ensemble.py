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
def test_crs_metadata_datum_ensemble():
    crs = pyproj.CRS('EPSG:4326')
    crs_json = crs.to_json_dict()
    check_ensemble = False
    if 'datum_ensemble' in crs_json:
        check_ensemble = True
        assert 'id' in crs_json['datum_ensemble']['members'][0]
    _remove_id_from_member_of_ensembles(crs_json)
    if check_ensemble:
        assert 'id' not in crs_json['datum_ensemble']['members'][0]
    assert pyproj.CRS(crs_json) == crs