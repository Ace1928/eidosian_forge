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
@pytest.mark.parametrize('conn', all_connectable_iris)
def test_read_sql_iris_no_parameter_with_percent(conn, request, sql_strings):
    if 'mysql' in conn or ('postgresql' in conn and 'adbc' not in conn):
        request.applymarker(pytest.mark.xfail(reason='broken test'))
    conn_name = conn
    conn = request.getfixturevalue(conn)
    query = sql_strings['read_no_parameters_with_percent'][flavor(conn_name)]
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            iris_frame = pandasSQL.read_query(query, params=None)
    check_iris_frame(iris_frame)