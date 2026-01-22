import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_write_postgis_append_when_table_exists(self, engine_postgis, df_nybb):
    """
        Tests that appending to existing table produces correct results when:
        if_replace='append'.
        """
    engine = engine_postgis
    table = 'nybb'
    orig_rows, orig_cols = df_nybb.shape
    write_postgis(df_nybb, con=engine, name=table, if_exists='replace')
    write_postgis(df_nybb, con=engine, name=table, if_exists='append')
    sql = text('SELECT * FROM {table};'.format(table=table))
    df = read_postgis(sql, engine, geom_col='geometry')
    new_rows, new_cols = df.shape
    assert new_rows == orig_rows * 2, ('There should be {target} rows,found: {current}'.format(target=orig_rows * 2, current=new_rows),)
    assert new_cols == orig_cols, ('There should be {target} columns,found: {current}'.format(target=orig_cols, current=new_cols),)