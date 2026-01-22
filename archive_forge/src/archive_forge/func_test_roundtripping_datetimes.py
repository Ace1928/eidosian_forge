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
def test_roundtripping_datetimes(sqlite_engine):
    conn = sqlite_engine
    df = DataFrame({'t': [datetime(2020, 12, 31, 12)]}, dtype='datetime64[ns]')
    df.to_sql('test', conn, if_exists='replace', index=False)
    result = pd.read_sql('select * from test', conn).iloc[0, 0]
    assert result == '2020-12-31 12:00:00.000000'