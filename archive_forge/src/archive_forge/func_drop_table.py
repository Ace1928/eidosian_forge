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
def drop_table(table_name: str, conn: sqlite3.Connection | sqlalchemy.engine.Engine | sqlalchemy.engine.Connection):
    if isinstance(conn, sqlite3.Connection):
        conn.execute(f'DROP TABLE IF EXISTS {sql._get_valid_sqlite_name(table_name)}')
        conn.commit()
    else:
        adbc = import_optional_dependency('adbc_driver_manager.dbapi', errors='ignore')
        if adbc and isinstance(conn, adbc.Connection):
            with conn.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        else:
            with conn.begin() as con:
                with sql.SQLDatabase(con) as db:
                    db.drop_table(table_name)