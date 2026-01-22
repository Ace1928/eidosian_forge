import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_write_postgis_with_esri_authority(self, engine_postgis, df_nybb):
    """
        Tests that GeoDataFrame can be written to PostGIS with ESRI Authority
        CRS information (GH #2414).
        """
    engine = engine_postgis
    table = 'nybb'
    df_nybb_esri = df_nybb.to_crs('ESRI:102003')
    write_postgis(df_nybb_esri, con=engine, name=table, if_exists='replace')
    sql = text("SELECT Find_SRID('{schema}', '{table}', '{geom_col}');".format(schema='public', table=table, geom_col='geometry'))
    with engine.connect() as conn:
        target_srid = conn.execute(sql).fetchone()[0]
    assert target_srid == 102003, 'SRID should be 102003, found %s' % target_srid