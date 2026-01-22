import contextlib
import json
from pathlib import Path
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.experimental.pandas as pd
from modin.config import AsyncReadMode, Engine
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import try_cast_to_pandas
@pytest.mark.skipif(Engine.get() not in ('Ray', 'Unidist', 'Dask'), reason=f'{Engine.get()} does not have experimental API')
def test_from_sql_defaults(tmp_path, make_sql_connection):
    filename = 'test_from_sql_distributed.db'
    table = 'test_from_sql_distributed'
    conn = make_sql_connection(str(tmp_path / filename), table)
    query = 'select * from {0}'.format(table)
    pandas_df = pandas.read_sql(query, conn)
    with pytest.warns(UserWarning):
        modin_df_from_query = pd.read_sql(query, conn)
    with pytest.warns(UserWarning):
        modin_df_from_table = pd.read_sql(table, conn)
    df_equals(modin_df_from_query, pandas_df)
    df_equals(modin_df_from_table, pandas_df)