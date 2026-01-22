import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
@pytest.mark.parametrize('dstype', ['fs', 'mem'])
def test_dataset_filter(tempdir, dstype):
    t1 = pa.table({'colA': [1, 2, 6, 8], 'col2': ['a', 'b', 'f', 'g']})
    if dstype == 'fs':
        ds.write_dataset(t1, tempdir / 't1', format='ipc')
        ds1 = ds.dataset(tempdir / 't1', format='ipc')
    elif dstype == 'mem':
        ds1 = ds.dataset(t1)
    else:
        raise NotImplementedError
    result = ds1.filter(pc.field('colA') < 3).filter(pc.field('col2') == 'a')
    expected = ds.FileSystemDataset if dstype == 'fs' else ds.InMemoryDataset
    assert isinstance(result, expected)
    assert result.to_table() == pa.table({'colA': [1], 'col2': ['a']})
    assert result.head(5) == pa.table({'colA': [1], 'col2': ['a']})
    r2 = ds1.filter(pc.field('colA') < 8).filter(pc.field('colA') > 1).scanner(filter=pc.field('colA') != 6)
    assert r2.to_table() == pa.table({'colA': [2], 'col2': ['b']})
    ds.write_dataset(result, tempdir / 'filtered', format='ipc')
    filtered = ds.dataset(tempdir / 'filtered', format='ipc')
    assert filtered.to_table() == pa.table({'colA': [1], 'col2': ['a']})
    joined = result.join(ds.dataset(pa.table({'colB': [10, 20], 'col2': ['a', 'b']})), keys='col2', join_type='right outer')
    assert joined.to_table().sort_by('colB') == pa.table({'colA': [1, None], 'colB': [10, 20], 'col2': ['a', 'b']})
    with pytest.raises(TypeError):
        ds1.filter(None)
    with pytest.raises(ValueError):
        result.get_fragments()
    schema_without_col2 = ds1.schema.remove(1)
    newschema = ds1.filter(pc.field('colA') < 3).replace_schema(schema_without_col2)
    assert newschema.to_table() == pa.table({'colA': [1, 2]})
    with pytest.raises(pa.ArrowInvalid):
        result.replace_schema(schema_without_col2).to_table()