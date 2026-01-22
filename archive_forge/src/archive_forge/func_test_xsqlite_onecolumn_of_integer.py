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
def test_xsqlite_onecolumn_of_integer(sqlite_buildin):
    mono_df = DataFrame([1, 2], columns=['c0'])
    assert sql.to_sql(mono_df, con=sqlite_buildin, name='mono_df', index=False) == 2
    con_x = sqlite_buildin
    the_sum = sum((my_c0[0] for my_c0 in con_x.execute('select * from mono_df')))
    assert the_sum == 3
    result = sql.read_sql('select * from mono_df', con_x)
    tm.assert_frame_equal(result, mono_df)