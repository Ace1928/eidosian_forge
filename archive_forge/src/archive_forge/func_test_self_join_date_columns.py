from __future__ import annotations
import contextlib
from contextlib import closing
import csv
from datetime import (
from io import StringIO
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING
import uuid
import numpy as np
import pytest
from pandas._libs import lib
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.util.version import Version
from pandas.io import sql
from pandas.io.sql import (
@pytest.mark.db
def test_self_join_date_columns(postgresql_psycopg2_engine):
    conn = postgresql_psycopg2_engine
    from sqlalchemy.sql import text
    create_table = text("\n    CREATE TABLE person\n    (\n        id serial constraint person_pkey primary key,\n        created_dt timestamp with time zone\n    );\n\n    INSERT INTO person\n        VALUES (1, '2021-01-01T00:00:00Z');\n    ")
    with conn.connect() as con:
        with con.begin():
            con.execute(create_table)
    sql_query = 'SELECT * FROM "person" AS p1 INNER JOIN "person" AS p2 ON p1.id = p2.id;'
    result = pd.read_sql(sql_query, conn)
    expected = DataFrame([[1, Timestamp('2021', tz='UTC')] * 2], columns=['id', 'created_dt'] * 2)
    tm.assert_frame_equal(result, expected)
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table('person')