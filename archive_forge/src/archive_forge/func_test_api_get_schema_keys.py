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
def test_api_get_schema_keys(conn, request, test_frame1):
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True))
    conn_name = conn
    conn = request.getfixturevalue(conn)
    frame = DataFrame({'Col1': [1.1, 1.2], 'Col2': [2.1, 2.2]})
    create_sql = sql.get_schema(frame, 'test', con=conn, keys='Col1')
    if 'mysql' in conn_name:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY (`Col1`)'
    else:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY ("Col1")'
    assert constraint_sentence in create_sql
    create_sql = sql.get_schema(test_frame1, 'test', con=conn, keys=['A', 'B'])
    if 'mysql' in conn_name:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY (`A`, `B`)'
    else:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY ("A", "B")'
    assert constraint_sentence in create_sql