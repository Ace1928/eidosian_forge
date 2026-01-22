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
def test_save_and_use(self):
    path = os.path.join(self.tmpdir, 'a')
    with FugueWorkflow() as dag:
        b = dag.df([[6, 1], [2, 7]], 'c:int,a:long')
        c = b.save_and_use(path, fmt='parquet')
        b.assert_eq(c)
    dag.run(self.engine)
    with FugueWorkflow() as dag:
        b = dag.df([[6, 1], [2, 7]], 'c:int,a:long')
        d = dag.load(path, fmt='parquet')
        b.assert_eq(d)
    dag.run(self.engine)