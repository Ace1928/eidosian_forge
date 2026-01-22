import datetime
import os
import pickle
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional
from unittest import TestCase
from uuid import uuid4
from triad.utils.io import write_text, join
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from fsspec.implementations.local import LocalFileSystem
from pytest import raises
from triad import SerializableRLock
import fugue.api as fa
from fugue import (
from fugue.column import col
from fugue.column import functions as ff
from fugue.column import lit
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.exceptions import (
@pytest.mark.skipif(os.name == 'nt', reason='Skip Windows')
@pytest.mark.skipif(not HAS_QPD, reason='qpd not working')
def test_any_column_name(self):
    f_parquet = os.path.join(str(self.tmpdir), 'a.parquet')
    f_csv = os.path.join(str(self.tmpdir), 'a.csv')

    def tr(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(**{'c *': 2})
    with fa.engine_context(self.engine):
        df1 = pd.DataFrame([[0, 1], [2, 3]], columns=['a b', ' '])
        df2 = pd.DataFrame([[0, 10], [20, 3]], columns=['a b', 'd'])
        r = fa.inner_join(df1, df2, as_fugue=True)
        df_eq(r, [[0, 1, 10]], '`a b`:long,` `:long,d:long', throw=True)
        r = fa.transform(r, tr)
        df_eq(r, [[0, 1, 10, 2]], '`a b`:long,` `:long,d:long,`c *`:long', throw=True)
        r = fa.alter_columns(r, '`c *`:str')
        r = fa.select(r, col('a b').alias('a b '), col(' ').alias('x y'), col('d'), col('c *').cast(int))
        df_eq(r, [[0, 1, 10, 2]], '`a b `:long,`x y`:long,d:long,`c *`:long', throw=True)
        r = fa.rename(r, {'a b ': 'a b'})
        fa.save(r, f_csv, header=True, force_single=True)
        fa.save(r, f_parquet)
        df_eq(fa.load(f_parquet, columns=['x y', 'd', 'c *'], as_fugue=True), [[1, 10, 2]], '`x y`:long,d:long,`c *`:long', throw=True)
        df_eq(fa.load(f_csv, header=True, infer_schema=False, columns=['d', 'c *'], as_fugue=True), [['10', '2']], 'd:str,`c *`:str', throw=True)
        df_eq(fa.load(f_csv, header=True, columns='`a b`:long,`x y`:long,d:long,`c *`:long', as_fugue=True), [[0, 1, 10, 2]], '`a b`:long,`x y`:long,d:long,`c *`:long', throw=True)
        r = fa.fugue_sql('\n                df1 = CREATE [[0, 1], [2, 3]] SCHEMA `a b`:long,` `:long\n                df2 = CREATE [[0, 10], [20, 3]] SCHEMA `a b`:long,d:long\n                SELECT df1.*,d FROM df1 INNER JOIN df2 ON df1.`a b`=df2.`a b`\n                ', as_fugue=True)
        df_eq(r, [[0, 1, 10]], '`a b`:long,` `:long,d:long', throw=True)
        r = fa.fugue_sql('\n                TRANSFORM r USING tr SCHEMA *,`c *`:long\n                ', as_fugue=True)
        df_eq(r, [[0, 1, 10, 2]], '`a b`:long,` `:long,d:long,`c *`:long', throw=True)
        r = fa.fugue_sql('\n                ALTER COLUMNS `c *`:long FROM r\n                ', as_fugue=True)
        df_eq(r, [[0, 1, 10, 2]], '`a b`:long,` `:long,d:long,`c *`:long', throw=True)
        res = fa.fugue_sql_flow('\n                LOAD "{{f_parquet}}" COLUMNS `x y`,d,`c *`\n                YIELD LOCAL DATAFRAME AS r1\n\n                LOAD "{{f_csv}}"(header=TRUE,infer_schema=FALSE) COLUMNS `x y`,d,`c *`\n                YIELD LOCAL DATAFRAME AS r2\n\n                LOAD "{{f_csv}}"(header=TRUE,infer_schema=FALSE)\n                COLUMNS `a b`:long,`x y`:long,d:long,`c *`:long\n                YIELD LOCAL DATAFRAME AS r3\n                ', f_parquet=f_parquet, f_csv=f_csv).run()
        df_eq(res['r1'], [[1, 10, 2]], '`x y`:long,d:long,`c *`:long', throw=True)
        df_eq(res['r2'], [['1', '10', '2']], '`x y`:str,d:str,`c *`:str', throw=True)
        df_eq(res['r3'], [[0, 1, 10, 2]], '`a b`:long,`x y`:long,d:long,`c *`:long', throw=True)