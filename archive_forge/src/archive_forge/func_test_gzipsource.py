from __future__ import absolute_import, print_function, division
import gzip
import bz2
import zipfile
from tempfile import NamedTemporaryFile
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
import petl as etl
from petl.io.sources import MemorySource, PopenSource, ZipSource, \
def test_gzipsource():
    tbl = [('foo', 'bar'), ('a', '1'), ('b', '2')]
    fn = NamedTemporaryFile().name + '.gz'
    expect = b'foo,bar\na,1\nb,2\n'
    etl.tocsv(tbl, GzipSource(fn), lineterminator='\n')
    actual = gzip.open(fn).read()
    eq_(expect, actual)
    etl.tocsv(tbl, fn, lineterminator='\n')
    actual = gzip.open(fn).read()
    eq_(expect, actual)
    tbl2 = etl.fromcsv(GzipSource(fn))
    ieq(tbl, tbl2)
    tbl2 = etl.fromcsv(fn)
    ieq(tbl, tbl2)