import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_read_postgis_null_geom(self, connection_spatialite, df_nybb):
    """Tests that geometry with NULL is accepted."""
    con = connection_spatialite
    geom_col = df_nybb.geometry.name
    df_nybb.geometry.iat[0] = None
    create_spatialite(con, df_nybb)
    sql = 'SELECT ogc_fid, borocode, boroname, shape_leng, shape_area, AsEWKB("{0}") AS "{0}" FROM nybb'.format(geom_col)
    df = read_postgis(sql, con, geom_col=geom_col)
    validate_boro_df(df)