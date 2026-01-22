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
def types_data():
    return [{'TextCol': 'first', 'DateCol': '2000-01-03 00:00:00', 'IntDateCol': 535852800, 'IntDateOnlyCol': 20101010, 'FloatCol': 10.1, 'IntCol': 1, 'BoolCol': False, 'IntColWithNull': 1, 'BoolColWithNull': False}, {'TextCol': 'first', 'DateCol': '2000-01-04 00:00:00', 'IntDateCol': 1356998400, 'IntDateOnlyCol': 20101212, 'FloatCol': 10.1, 'IntCol': 1, 'BoolCol': False, 'IntColWithNull': None, 'BoolColWithNull': None}]