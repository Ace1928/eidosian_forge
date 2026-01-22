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
def test_api_complex_raises(conn, request):
    conn_name = conn
    conn = request.getfixturevalue(conn)
    df = DataFrame({'a': [1 + 1j, 2j]})
    if 'adbc' in conn_name:
        msg = 'datatypes not supported'
    else:
        msg = 'Complex datatypes not supported'
    with pytest.raises(ValueError, match=msg):
        assert df.to_sql('test_complex', con=conn) is None