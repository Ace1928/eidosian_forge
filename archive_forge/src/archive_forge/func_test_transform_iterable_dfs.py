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
def test_transform_iterable_dfs(self):

    def mt_pandas(dfs: Iterable[pd.DataFrame], empty: bool=False) -> Iterator[pd.DataFrame]:
        for df in dfs:
            if not empty:
                df = df.assign(c=2)
                df = df[reversed(list(df.columns))]
                yield df
    with FugueWorkflow() as dag:
        a = dag.df([[1, 2], [3, 4]], 'a:int,b:int')
        b = a.transform(mt_pandas)
        dag.df([[1, 2, 2], [3, 4, 2]], 'a:int,b:int,c:int').assert_eq(b)
    dag.run(self.engine)
    with FugueWorkflow() as dag:
        a = dag.df([[1, 2], [3, 4]], 'a:int,b:int')
        b = a.transform(mt_pandas, params=dict(empty=True))
        dag.df([], 'a:int,b:int,c:int').assert_eq(b)
        b = a.partition_by('a').transform(mt_pandas, params=dict(empty=True))
        dag.df([], 'a:int,b:int,c:int').assert_eq(b)
    dag.run(self.engine)

    def mt_arrow(dfs: Iterable[pa.Table], empty: bool=False) -> Iterator[pa.Table]:
        for df in dfs:
            if not empty:
                df = df.select(reversed(df.schema.names))
                yield df

    def mt_arrow_2(dfs: Iterable[pa.Table]) -> Iterator[pa.Table]:
        for df in dfs:
            yield df.drop(['b'])
    with FugueWorkflow() as dag:
        a = dag.df([[1, 2], [3, 4]], 'a:int,b:int')
        b = a.transform(mt_arrow)
        dag.df([[1, 2], [3, 4]], 'a:int,b:int').assert_eq(b)
        b = a.transform(mt_arrow_2)
        dag.df([[1], [3]], 'a:long').assert_eq(b)
    dag.run(self.engine)
    with FugueWorkflow() as dag:
        a = dag.df([[1, 2], [3, 4]], 'a:int,b:int')
        b = a.transform(mt_arrow, params=dict(empty=True))
        dag.df([], 'a:int,b:int').assert_eq(b)
        b = a.partition_by('a').transform(mt_arrow, params=dict(empty=True))
        dag.df([], 'a:int,b:int').assert_eq(b)
    dag.run(self.engine)