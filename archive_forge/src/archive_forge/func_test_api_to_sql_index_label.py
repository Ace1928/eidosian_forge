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
@pytest.mark.parametrize('conn', all_connectable)
@pytest.mark.parametrize('index_name,index_label,expected', [(None, None, 'index'), (None, 'other_label', 'other_label'), ('index_name', None, 'index_name'), ('index_name', 'other_label', 'other_label'), (0, None, '0'), (None, 0, '0')])
def test_api_to_sql_index_label(conn, request, index_name, index_label, expected):
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason='index_label argument NotImplemented with ADBC'))
    conn = request.getfixturevalue(conn)
    if sql.has_table('test_index_label', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_index_label')
    temp_frame = DataFrame({'col1': range(4)})
    temp_frame.index.name = index_name
    query = 'SELECT * FROM test_index_label'
    sql.to_sql(temp_frame, 'test_index_label', conn, index_label=index_label)
    frame = sql.read_sql_query(query, conn)
    assert frame.columns[0] == expected