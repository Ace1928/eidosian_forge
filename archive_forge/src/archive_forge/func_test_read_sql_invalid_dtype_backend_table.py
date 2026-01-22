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
@pytest.mark.parametrize('func', ['read_sql', 'read_sql_table', 'read_sql_query'])
def test_read_sql_invalid_dtype_backend_table(conn, request, func, dtype_backend_data):
    conn = request.getfixturevalue(conn)
    table = 'test'
    df = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists='replace')
    msg = "dtype_backend numpy is invalid, only 'numpy_nullable' and 'pyarrow' are allowed."
    with pytest.raises(ValueError, match=msg):
        getattr(pd, func)(table, conn, dtype_backend='numpy')