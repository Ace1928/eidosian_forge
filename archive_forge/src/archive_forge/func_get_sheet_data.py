from __future__ import annotations
from datetime import (
from typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
import pandas as pd
from pandas.core.shared_docs import _shared_docs
from pandas.io.excel._base import BaseExcelReader
def get_sheet_data(self, sheet: CalamineSheet, file_rows_needed: int | None=None) -> list[list[Scalar | NaTType | time]]:

    def _convert_cell(value: _CellValue) -> Scalar | NaTType | time:
        if isinstance(value, float):
            val = int(value)
            if val == value:
                return val
            else:
                return value
        elif isinstance(value, date):
            return pd.Timestamp(value)
        elif isinstance(value, timedelta):
            return pd.Timedelta(value)
        elif isinstance(value, time):
            return value
        return value
    rows: list[list[_CellValue]] = sheet.to_python(skip_empty_area=False, nrows=file_rows_needed)
    data = [[_convert_cell(cell) for cell in row] for row in rows]
    return data