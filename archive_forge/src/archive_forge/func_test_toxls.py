from __future__ import division, print_function, absolute_import
from datetime import datetime
from tempfile import NamedTemporaryFile
import pytest
import petl as etl
from petl.io.xls import fromxls, toxls
from petl.test.helpers import ieq
def test_toxls():
    expect = (('foo', 'bar'), ('A', 1), ('B', 2), ('C', 2))
    f = NamedTemporaryFile(delete=False)
    f.close()
    toxls(expect, f.name, 'Sheet1')
    actual = fromxls(f.name, 'Sheet1')
    ieq(expect, actual)
    ieq(expect, actual)