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
@pytest.fixture
def test_frame3():
    columns = ['index', 'A', 'B']
    data = [('2000-01-03 00:00:00', 2 ** 31 - 1, -1.98767), ('2000-01-04 00:00:00', -29, -0.0412318367011), ('2000-01-05 00:00:00', 20000, 0.731167677815), ('2000-01-06 00:00:00', -290867, 1.56762092543)]
    return DataFrame(data, columns=columns)