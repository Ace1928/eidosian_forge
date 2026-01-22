from __future__ import absolute_import, print_function, division
import gzip
import bz2
import zipfile
from tempfile import NamedTemporaryFile
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
import petl as etl
from petl.io.sources import MemorySource, PopenSource, ZipSource, \
def test_stdoutsource_unicode():
    tbl = [('foo', 'bar'), (u'Արամ Խաչատրյան', 1), (u'Johann Strauß', 2)]
    etl.tocsv(tbl, StdoutSource(), encoding='utf-8')
    etl.tohtml(tbl, StdoutSource(), encoding='utf-8')
    etl.topickle(tbl, StdoutSource())