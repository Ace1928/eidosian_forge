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
def test_sqlite_read_sql_delegate(sqlite_buildin_iris):
    conn = sqlite_buildin_iris
    iris_frame1 = sql.read_sql_query('SELECT * FROM iris', conn)
    iris_frame2 = sql.read_sql('SELECT * FROM iris', conn)
    tm.assert_frame_equal(iris_frame1, iris_frame2)
    msg = 'Execution failed on sql \'iris\': near "iris": syntax error'
    with pytest.raises(sql.DatabaseError, match=msg):
        sql.read_sql('iris', conn)