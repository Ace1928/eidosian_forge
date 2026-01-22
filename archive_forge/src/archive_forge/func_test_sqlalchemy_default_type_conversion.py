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
@pytest.mark.parametrize('conn', sqlalchemy_connectable_types)
def test_sqlalchemy_default_type_conversion(conn, request):
    conn_name = conn
    if conn_name == 'sqlite_str':
        pytest.skip('types tables not created in sqlite_str fixture')
    elif 'mysql' in conn_name or 'sqlite' in conn_name:
        request.applymarker(pytest.mark.xfail(reason='boolean dtype not inferred properly'))
    conn = request.getfixturevalue(conn)
    df = sql.read_sql_table('types', conn)
    assert issubclass(df.FloatCol.dtype.type, np.floating)
    assert issubclass(df.IntCol.dtype.type, np.integer)
    assert issubclass(df.BoolCol.dtype.type, np.bool_)
    assert issubclass(df.IntColWithNull.dtype.type, np.floating)
    assert issubclass(df.BoolColWithNull.dtype.type, object)