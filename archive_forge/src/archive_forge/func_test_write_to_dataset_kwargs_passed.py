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
@pytest.mark.parametrize('write_dataset_kwarg', (('create_dir', True), ('create_dir', False)))
def test_write_to_dataset_kwargs_passed(tempdir, write_dataset_kwarg):
    """Verify kwargs in pq.write_to_dataset are passed onto ds.write_dataset"""
    import pyarrow.dataset as ds
    table = pa.table({'a': [1, 2, 3]})
    path = tempdir / 'out.parquet'
    signature = inspect.signature(ds.write_dataset)
    key, arg = write_dataset_kwarg
    assert key not in inspect.signature(pq.write_to_dataset).parameters
    assert key in signature.parameters
    with mock.patch.object(ds, 'write_dataset', autospec=True) as mock_write_dataset:
        pq.write_to_dataset(table, path, **{key: arg})
        _name, _args, kwargs = mock_write_dataset.mock_calls[0]
        assert kwargs[key] == arg