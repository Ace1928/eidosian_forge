from __future__ import absolute_import, print_function, division
from datetime import datetime
from tempfile import NamedTemporaryFile
import pytest
import petl as etl
from petl.io.xlsx import fromxlsx, toxlsx, appendxlsx
from petl.test.helpers import ieq, eq_
def test_toxlsx_add_sheet_match(xlsx_test_table):
    tbl = xlsx_test_table
    f = NamedTemporaryFile(delete=True, suffix='.xlsx')
    f.close()
    toxlsx(tbl, f.name, 'Sheet1', mode='overwrite')
    with pytest.raises(ValueError) as excinfo:
        toxlsx(tbl, f.name, 'Sheet1', mode='add')
    assert 'Sheet Sheet1 already exists in file' in str(excinfo.value)