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
@pytest.mark.parametrize('format,schema_version', product(['feather', 'parquet'], [None] + SUPPORTED_VERSIONS))
def test_write_spec_version(tmpdir, format, schema_version):
    if format == 'feather':
        from pyarrow.feather import read_table
    else:
        from pyarrow.parquet import read_table
    filename = os.path.join(str(tmpdir), f'test.{format}')
    gdf = geopandas.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs='EPSG:4326')
    write = getattr(gdf, f'to_{format}')
    write(filename, schema_version=schema_version)
    read = getattr(geopandas, f'read_{format}')
    df = read(filename)
    assert_geodataframe_equal(df, gdf)
    schema_version = schema_version or METADATA_VERSION
    table = read_table(filename)
    metadata = json.loads(table.schema.metadata[b'geo'])
    assert metadata['version'] == schema_version
    if schema_version == '0.1.0':
        assert metadata['columns']['geometry']['crs'] == gdf.crs.to_wkt()
    else:
        crs_expected = gdf.crs.to_json_dict()
        _remove_id_from_member_of_ensembles(crs_expected)
        assert metadata['columns']['geometry']['crs'] == crs_expected
    if Version(schema_version) <= Version('0.4.0'):
        assert 'geometry_type' in metadata['columns']['geometry']
        assert metadata['columns']['geometry']['geometry_type'] == 'Polygon'
    else:
        assert 'geometry_types' in metadata['columns']['geometry']
        assert metadata['columns']['geometry']['geometry_types'] == ['Polygon']