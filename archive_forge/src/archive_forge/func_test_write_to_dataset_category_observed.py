import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_write_to_dataset_category_observed(tempdir):
    df = pd.DataFrame({'cat': pd.Categorical(['a', 'b', 'a'], categories=['a', 'b', 'c']), 'col': [1, 2, 3]})
    table = pa.table(df)
    path = tempdir / 'dataset'
    pq.write_to_dataset(table, tempdir / 'dataset', partition_cols=['cat'])
    subdirs = [f.name for f in path.iterdir() if f.is_dir()]
    assert len(subdirs) == 2
    assert 'cat=c' not in subdirs