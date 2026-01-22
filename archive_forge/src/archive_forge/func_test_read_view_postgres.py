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
@pytest.mark.parametrize('conn', postgresql_connectable)
def test_read_view_postgres(conn, request):
    conn = request.getfixturevalue(conn)
    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text
    table_name = f'group_{uuid.uuid4().hex}'
    view_name = f'group_view_{uuid.uuid4().hex}'
    sql_stmt = text(f"\n    CREATE TABLE {table_name} (\n        group_id INTEGER,\n        name TEXT\n    );\n    INSERT INTO {table_name} VALUES\n        (1, 'name');\n    CREATE VIEW {view_name}\n    AS\n    SELECT * FROM {table_name};\n    ")
    if isinstance(conn, Engine):
        with conn.connect() as con:
            with con.begin():
                con.execute(sql_stmt)
    else:
        with conn.begin():
            conn.execute(sql_stmt)
    result = read_sql_table(view_name, conn)
    expected = DataFrame({'group_id': [1], 'name': 'name'})
    tm.assert_frame_equal(result, expected)