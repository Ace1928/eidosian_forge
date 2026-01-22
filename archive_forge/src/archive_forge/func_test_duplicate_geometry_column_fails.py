import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
@pytest.mark.xfail(compat.PANDAS_GE_20 and (not compat.PANDAS_GE_21), reason='Duplicate columns are dropped in read_sql with pandas 2.0.x')
def test_duplicate_geometry_column_fails(self, engine_postgis):
    """
        Tests that a ValueError is raised if an SQL query returns two geometry columns.
        """
    engine = engine_postgis
    sql = 'select ST_MakePoint(0, 0) as geom, ST_MakePoint(0, 0) as geom;'
    with pytest.raises(ValueError):
        read_postgis(sql, engine, geom_col='geom')