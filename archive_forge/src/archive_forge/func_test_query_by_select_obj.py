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
@pytest.mark.parametrize('conn', sqlalchemy_connectable_iris)
def test_query_by_select_obj(conn, request):
    conn = request.getfixturevalue(conn)
    from sqlalchemy import bindparam, select
    iris = iris_table_metadata()
    name_select = select(iris).where(iris.c.Name == bindparam('name'))
    iris_df = sql.read_sql(name_select, conn, params={'name': 'Iris-setosa'})
    all_names = set(iris_df['Name'])
    assert all_names == {'Iris-setosa'}