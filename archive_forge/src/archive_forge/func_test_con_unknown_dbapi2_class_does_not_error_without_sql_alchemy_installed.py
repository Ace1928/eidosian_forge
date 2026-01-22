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
@td.skip_if_installed('sqlalchemy')
def test_con_unknown_dbapi2_class_does_not_error_without_sql_alchemy_installed():

    class MockSqliteConnection:

        def __init__(self, *args, **kwargs) -> None:
            self.conn = sqlite3.Connection(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def close(self):
            self.conn.close()
    with contextlib.closing(MockSqliteConnection(':memory:')) as conn:
        with tm.assert_produces_warning(UserWarning):
            sql.read_sql('SELECT 1', conn)