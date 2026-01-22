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
def test_xsqlite_write_row_by_row(sqlite_buildin):
    frame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    frame.iloc[0, 0] = np.nan
    create_sql = sql.get_schema(frame, 'test')
    cur = sqlite_buildin.cursor()
    cur.execute(create_sql)
    ins = 'INSERT INTO test VALUES (%s, %s, %s, %s)'
    for _, row in frame.iterrows():
        fmt_sql = format_query(ins, *row)
        tquery(fmt_sql, con=sqlite_buildin)
    sqlite_buildin.commit()
    result = sql.read_sql('select * from test', con=sqlite_buildin)
    result.index = frame.index
    tm.assert_frame_equal(result, frame, rtol=0.001)