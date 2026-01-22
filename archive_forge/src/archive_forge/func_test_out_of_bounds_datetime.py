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
def test_out_of_bounds_datetime(conn, request):
    conn = request.getfixturevalue(conn)
    data = DataFrame({'date': datetime(9999, 1, 1)}, index=[0])
    assert data.to_sql(name='test_datetime_obb', con=conn, index=False) == 1
    result = sql.read_sql_table('test_datetime_obb', conn)
    expected = DataFrame([pd.NaT], columns=['date'])
    tm.assert_frame_equal(result, expected)