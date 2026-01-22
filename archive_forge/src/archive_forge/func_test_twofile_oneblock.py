from __future__ import annotations
import os
import random
import pytest
import dask.bag as db
def test_twofile_oneblock(tmpdir):
    tmpdir = str(tmpdir)
    fn1 = os.path.join(tmpdir, 'one.avro')
    fn2 = os.path.join(tmpdir, 'two.avro')
    with open(fn1, 'wb') as f:
        fastavro.writer(f, records=expected[:500], schema=schema)
    with open(fn2, 'wb') as f:
        fastavro.writer(f, records=expected[500:], schema=schema)
    b = db.read_avro(os.path.join(tmpdir, '*.avro'), blocksize=None)
    assert b.npartitions == 2
    assert b.compute() == expected