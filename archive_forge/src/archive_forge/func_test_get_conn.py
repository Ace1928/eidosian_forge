import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_get_conn(self, engine_postgis):
    Connection = pytest.importorskip('sqlalchemy.engine.base').Connection
    engine = engine_postgis
    with get_conn(engine) as output:
        assert isinstance(output, Connection)
    with engine.connect() as conn:
        with get_conn(conn) as output:
            assert isinstance(output, Connection)
    with pytest.raises(ValueError):
        with get_conn(object()):
            pass