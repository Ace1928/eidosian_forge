from __future__ import division, print_function, absolute_import
from itertools import chain
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.transform.sorts import sort
import petl as etl
from petl.io.pytables import fromhdf5, fromhdf5sorted, tohdf5, appendhdf5
def test_tohdf5_create():
    table1 = (('foo', 'bar'), (1, b'asdfgh'), (2, b'qwerty'), (3, b'zxcvbn'))
    f = NamedTemporaryFile()
    tohdf5(table1, f.name, '/testgroup', 'testtable', create=True, drop=True, description=FooBar, createparents=True)
    ieq(table1, fromhdf5(f.name, '/testgroup', 'testtable'))
    tohdf5(table1, f.name, '/testgroup', 'testtable2', create=True, drop=True, createparents=True)
    ieq(table1, fromhdf5(f.name, '/testgroup', 'testtable2'))