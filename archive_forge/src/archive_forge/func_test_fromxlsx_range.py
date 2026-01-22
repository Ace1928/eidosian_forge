from __future__ import absolute_import, print_function, division
from datetime import datetime
from tempfile import NamedTemporaryFile
import pytest
import petl as etl
from petl.io.xlsx import fromxlsx, toxlsx, appendxlsx
from petl.test.helpers import ieq, eq_
def test_fromxlsx_range(xlsx_test_filename):
    tbl = fromxlsx(xlsx_test_filename, 'Sheet2', range_string='B2:C6')
    expect = (('foo', 'bar'), ('A', 1), ('B', 2), ('C', 2), (u'é', datetime(2012, 1, 1)))
    ieq(expect, tbl)
    ieq(expect, tbl)