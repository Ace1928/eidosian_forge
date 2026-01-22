from datetime import (
import re
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
import pandas._testing as tm
from pandas.io.excel import ExcelWriter
@pytest.mark.parametrize(['value', 'cell_value_type', 'cell_value_attribute', 'cell_value'], argvalues=[(True, 'boolean', 'boolean-value', 'true'), ('test string', 'string', 'string-value', 'test string'), (1, 'float', 'value', '1'), (1.5, 'float', 'value', '1.5'), (datetime(2010, 10, 10, 10, 10, 10), 'date', 'date-value', '2010-10-10T10:10:10'), (date(2010, 10, 10), 'date', 'date-value', '2010-10-10')])
def test_cell_value_type(ext, value, cell_value_type, cell_value_attribute, cell_value):
    from odf.namespaces import OFFICENS
    from odf.table import TableCell, TableRow
    table_cell_name = TableCell().qname
    with tm.ensure_clean(ext) as f:
        pd.DataFrame([[value]]).to_excel(f, header=False, index=False)
        with pd.ExcelFile(f) as wb:
            sheet = wb._reader.get_sheet_by_index(0)
            sheet_rows = sheet.getElementsByType(TableRow)
            sheet_cells = [x for x in sheet_rows[0].childNodes if hasattr(x, 'qname') and x.qname == table_cell_name]
            cell = sheet_cells[0]
            assert cell.attributes.get((OFFICENS, 'value-type')) == cell_value_type
            assert cell.attributes.get((OFFICENS, cell_value_attribute)) == cell_value