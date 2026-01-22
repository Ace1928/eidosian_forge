from __future__ import absolute_import, print_function, division
from datetime import datetime
from tempfile import NamedTemporaryFile
import pytest
import petl as etl
from petl.io.xlsx import fromxlsx, toxlsx, appendxlsx
from petl.test.helpers import ieq, eq_
def test_toxlsx_replace_sheet_nofile(xlsx_test_table):
    f = NamedTemporaryFile(delete=True, suffix='.xlsx')
    f.close()
    toxlsx(xlsx_test_table, f.name, 'Sheet1', mode='replace')
    wb = openpyxl.load_workbook(f.name, read_only=True)
    eq_(1, len(wb.sheetnames))