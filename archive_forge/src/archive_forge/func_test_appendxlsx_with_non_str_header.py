from __future__ import absolute_import, print_function, division
from datetime import datetime
from tempfile import NamedTemporaryFile
import pytest
import petl as etl
from petl.io.xlsx import fromxlsx, toxlsx, appendxlsx
from petl.test.helpers import ieq, eq_
def test_appendxlsx_with_non_str_header(xlsx_table_with_non_str_header, xlsx_test_table):
    f = NamedTemporaryFile(delete=True, suffix='.xlsx')
    f.close()
    toxlsx(xlsx_test_table, f.name, 'Sheet1')
    actual = fromxlsx(f.name, 'Sheet1')
    ieq(xlsx_test_table, actual)
    appendxlsx(xlsx_table_with_non_str_header, f.name, 'Sheet1')
    expect = etl.cat(xlsx_test_table, xlsx_table_with_non_str_header)
    ieq(expect, actual)