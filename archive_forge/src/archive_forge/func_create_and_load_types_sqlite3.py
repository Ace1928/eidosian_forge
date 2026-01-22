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
def create_and_load_types_sqlite3(conn, types_data: list[dict]):
    stmt = 'CREATE TABLE types (\n                    "TextCol" TEXT,\n                    "DateCol" TEXT,\n                    "IntDateCol" INTEGER,\n                    "IntDateOnlyCol" INTEGER,\n                    "FloatCol" REAL,\n                    "IntCol" INTEGER,\n                    "BoolCol" INTEGER,\n                    "IntColWithNull" INTEGER,\n                    "BoolColWithNull" INTEGER\n                )'
    ins_stmt = '\n                INSERT INTO types\n                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)\n                '
    if isinstance(conn, sqlite3.Connection):
        cur = conn.cursor()
        cur.execute(stmt)
        cur.executemany(ins_stmt, types_data)
    else:
        with conn.cursor() as cur:
            cur.execute(stmt)
            cur.executemany(ins_stmt, types_data)
        conn.commit()