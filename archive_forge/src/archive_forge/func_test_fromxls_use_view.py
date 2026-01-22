from __future__ import division, print_function, absolute_import
from datetime import datetime
from tempfile import NamedTemporaryFile
import pytest
import petl as etl
from petl.io.xls import fromxls, toxls
from petl.test.helpers import ieq
def test_fromxls_use_view():
    filename = _get_test_xls()
    if filename is None:
        return
    tbl = fromxls(filename, 'Sheet1', use_view=False)
    expect = (('foo', 'bar'), ('A', 1), ('B', 2), ('C', 2), (u'é', 40909.0))
    ieq(expect, tbl)
    ieq(expect, tbl)