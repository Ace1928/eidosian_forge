from __future__ import absolute_import, print_function, division
import gzip
import bz2
import zipfile
from tempfile import NamedTemporaryFile
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
import petl as etl
from petl.io.sources import MemorySource, PopenSource, ZipSource, \
def test_memorysource_2():
    data = 'foo,bar\r\na,1\r\nb,2\r\nc,2\r\n'
    if not PY2:
        data = data.encode('ascii')
    actual = etl.fromcsv(MemorySource(data))
    expect = (('foo', 'bar'), ('a', '1'), ('b', '2'), ('c', '2'))
    ieq(expect, actual)
    ieq(expect, actual)