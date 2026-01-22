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
def test_create_process_output(self):
    with FugueWorkflow() as dag:
        a = dag.create(mock_creator, params=dict(p=2))
        a.assert_eq(ArrayDataFrame([[2]], 'a:int'))
        b = dag.process(a, a, using=mock_processor)
        b.assert_eq(ArrayDataFrame([[2]], 'a:int'))
        b = dag.process(dict(df1=a, df2=a), using=mock_processor)
        b.assert_eq(ArrayDataFrame([[2]], 'a:int'))
        dag.output(a, b, using=mock_outputter)
        b2 = dag.process(a, a, a, using=mock_processor2)
        b2.assert_eq(ArrayDataFrame([[3]], 'a:int'))
        b2 = dag.process(a, a, a, using=MockProcessor3)
        b2.assert_eq(ArrayDataFrame([[3]], 'a:int'))
        a.process(mock_processor2).assert_eq(ArrayDataFrame([[1]], 'a:int'))
        a.output(mock_outputter2)
        dag.output(dict(df=a), using=mock_outputter2)
        a.partition(num=3).output(MockOutputter3)
        dag.output(dict(aa=a, bb=b), using=MockOutputter4)
        a = dag.create(mock_creator2, params=dict(p=2))
        b = dag.create(mock_creator2, params=dict(p=2))
        c = dag.process(a, b, using=mock_processor4)
        c.assert_eq(ArrayDataFrame([[2]], 'a:int'))
        dag.output(a, b, using=mock_outputter4)
    dag.run(self.engine)