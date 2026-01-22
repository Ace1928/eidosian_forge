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
def test_con_string_import_error():
    conn = 'mysql://root@localhost/pandas'
    msg = 'Using URI string without sqlalchemy installed'
    with pytest.raises(ImportError, match=msg):
        sql.read_sql('SELECT * FROM iris', conn)