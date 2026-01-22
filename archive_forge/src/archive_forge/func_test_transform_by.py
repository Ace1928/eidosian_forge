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
def test_transform_by(self):
    with FugueWorkflow() as dag:
        a = dag.df([[1, 2], [None, 1], [3, 4], [None, 4]], 'a:double,b:int')
        c = a.transform(MockTransform1, pre_partition={'by': ['a']})
        dag.df([[None, 1, 2, 1], [None, 4, 2, 1], [1, 2, 1, 1], [3, 4, 1, 1]], 'a:double,b:int,ct:int,p:int').assert_eq(c)
        c = a.transform(mock_tf1, pre_partition={'by': ['a'], 'presort': 'b DESC'})
        dag.df([[None, 4, 2, 1], [None, 1, 2, 1], [1, 2, 1, 1], [3, 4, 1, 1]], 'a:double,b:int,ct:int,p:int').assert_eq(c)
        c = a.transform(mock_tf2_except, schema='*', pre_partition={'by': ['a'], 'presort': 'b DESC'}, ignore_errors=[NotImplementedError])
        dag.df([[1, 2], [3, 4]], 'a:double,b:int').assert_eq(c)
        c = a.partition(by='a', presort='b DESC').transform(mock_tf2_except, schema='*', ignore_errors=[NotImplementedError])
        dag.df([[1, 2], [3, 4]], 'a:double,b:int').assert_eq(c)
    dag.run(self.engine)