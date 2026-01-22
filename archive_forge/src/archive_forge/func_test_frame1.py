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
def test_frame1():
    columns = ['index', 'A', 'B', 'C', 'D']
    data = [('2000-01-03 00:00:00', 0.980268513777, 3.68573087906, -0.364216805298, -1.15973806169), ('2000-01-04 00:00:00', 1.04791624281, -0.0412318367011, -0.16181208307, 0.212549316967), ('2000-01-05 00:00:00', 0.498580885705, 0.731167677815, -0.537677223318, 1.34627041952), ('2000-01-06 00:00:00', 1.12020151869, 1.56762092543, 0.00364077397681, 0.67525259227)]
    return DataFrame(data, columns=columns)