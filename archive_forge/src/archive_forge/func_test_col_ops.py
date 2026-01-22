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
def test_col_ops(self):
    with FugueWorkflow() as dag:
        a = dag.df([[1, 10], [2, 20]], 'x:long,y:long')
        aa = dag.df([[1, 10], [2, 20]], 'xx:long,y:long')
        a.rename({'x': 'xx'}).assert_eq(aa)
        a[['x']].assert_eq(ArrayDataFrame([[1], [2]], 'x:long'))
        a.drop(['y', 'yy'], if_exists=True).assert_eq(ArrayDataFrame([[1], [2]], 'x:long'))
        a[['x']].rename(x='xx').assert_eq(ArrayDataFrame([[1], [2]], 'xx:long'))
        a.alter_columns('x:str').assert_eq(ArrayDataFrame([['1', 10], ['2', 20]], 'x:str,y:long'))
    dag.run(self.engine)