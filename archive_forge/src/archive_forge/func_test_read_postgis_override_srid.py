import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_read_postgis_override_srid(self, connection_postgis, df_nybb):
    """Tests that a user specified CRS overrides the geodatabase SRID."""
    con = connection_postgis
    orig_crs = df_nybb.crs
    create_postgis(con, df_nybb, srid=4269)
    sql = 'SELECT * FROM nybb;'
    df = read_postgis(sql, con, crs=orig_crs)
    validate_boro_df(df)
    assert df.crs == orig_crs