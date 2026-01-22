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
def test_sqlalchemy_type_mapping(conn, request):
    conn = request.getfixturevalue(conn)
    from sqlalchemy import TIMESTAMP
    df = DataFrame({'time': to_datetime(['2014-12-12 01:54', '2014-12-11 02:54'], utc=True)})
    with sql.SQLDatabase(conn) as db:
        table = sql.SQLTable('test_type', db, frame=df)
        assert isinstance(table.table.c['time'].type, TIMESTAMP)