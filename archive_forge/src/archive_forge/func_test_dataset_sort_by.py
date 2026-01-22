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
def test_dataset_sort_by(tempdir, dstype):
    table = pa.table([pa.array([3, 1, 4, 2, 5]), pa.array(['b', 'a', 'b', 'a', 'c'])], names=['values', 'keys'])
    if dstype == 'fs':
        ds.write_dataset(table, tempdir / 't1', format='ipc')
        dt = ds.dataset(tempdir / 't1', format='ipc')
    elif dstype == 'mem':
        dt = ds.dataset(table)
    else:
        raise NotImplementedError
    assert dt.sort_by('values').to_table().to_pydict() == {'keys': ['a', 'a', 'b', 'b', 'c'], 'values': [1, 2, 3, 4, 5]}
    assert dt.sort_by([('values', 'descending')]).to_table().to_pydict() == {'keys': ['c', 'b', 'b', 'a', 'a'], 'values': [5, 4, 3, 2, 1]}
    assert dt.filter(pc.field('values') < 4).sort_by('values').to_table().to_pydict() == {'keys': ['a', 'a', 'b'], 'values': [1, 2, 3]}
    table = pa.Table.from_arrays([pa.array([5, 7, 7, 35], type=pa.int64()), pa.array(['foo', 'car', 'bar', 'foobar'])], names=['a', 'b'])
    dt = ds.dataset(table)
    sorted_tab = dt.sort_by([('a', 'descending')])
    sorted_tab_dict = sorted_tab.to_table().to_pydict()
    assert sorted_tab_dict['a'] == [35, 7, 7, 5]
    assert sorted_tab_dict['b'] == ['foobar', 'car', 'bar', 'foo']
    sorted_tab = dt.sort_by([('a', 'ascending')])
    sorted_tab_dict = sorted_tab.to_table().to_pydict()
    assert sorted_tab_dict['a'] == [5, 7, 7, 35]
    assert sorted_tab_dict['b'] == ['foo', 'car', 'bar', 'foobar']