from __future__ import absolute_import, print_function, division
import gzip
import bz2
import zipfile
from tempfile import NamedTemporaryFile
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
import petl as etl
from petl.io.sources import MemorySource, PopenSource, ZipSource, \
def test_popensource():
    expect = (('foo', 'bar'),)
    delimiter = ' '
    actual = etl.fromcsv(PopenSource('echo foo bar', shell=True), delimiter=delimiter)
    ieq(expect, actual)