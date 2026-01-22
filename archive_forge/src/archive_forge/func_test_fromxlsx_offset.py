from __future__ import absolute_import, print_function, division
from datetime import datetime
from tempfile import NamedTemporaryFile
import pytest
import petl as etl
from petl.io.xlsx import fromxlsx, toxlsx, appendxlsx
from petl.test.helpers import ieq, eq_
def test_fromxlsx_offset(xlsx_test_filename):
    tbl = fromxlsx(xlsx_test_filename, 'Sheet1', min_row=2, min_col=2)
    expect = ((1,), (2,), (2,), (datetime(2012, 1, 1, 0, 0),))
    ieq(expect, tbl)
    ieq(expect, tbl)