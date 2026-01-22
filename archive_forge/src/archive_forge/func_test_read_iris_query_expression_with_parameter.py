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
@pytest.mark.parametrize('conn', sqlalchemy_connectable_iris)
def test_read_iris_query_expression_with_parameter(conn, request):
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'chunksize' not implemented for ADBC drivers", strict=True))
    conn = request.getfixturevalue(conn)
    from sqlalchemy import MetaData, Table, create_engine, select
    metadata = MetaData()
    autoload_con = create_engine(conn) if isinstance(conn, str) else conn
    iris = Table('iris', metadata, autoload_with=autoload_con)
    iris_frame = read_sql_query(select(iris), conn, params={'name': 'Iris-setosa', 'length': 5.1})
    check_iris_frame(iris_frame)
    if isinstance(conn, str):
        autoload_con.dispose()