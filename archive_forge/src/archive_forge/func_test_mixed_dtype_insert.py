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
@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_mixed_dtype_insert(conn, request):
    conn = request.getfixturevalue(conn)
    s1 = Series(2 ** 25 + 1, dtype=np.int32)
    s2 = Series(0.0, dtype=np.float32)
    df = DataFrame({'s1': s1, 's2': s2})
    assert df.to_sql(name='test_read_write', con=conn, index=False) == 1
    df2 = sql.read_sql_table('test_read_write', conn)
    tm.assert_frame_equal(df, df2, check_dtype=False, check_exact=True)