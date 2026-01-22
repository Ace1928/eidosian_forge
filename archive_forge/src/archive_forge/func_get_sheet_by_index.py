from __future__ import annotations
from datetime import (
from typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
import pandas as pd
from pandas.core.shared_docs import _shared_docs
from pandas.io.excel._base import BaseExcelReader
def get_sheet_by_index(self, index: int) -> CalamineSheet:
    self.raise_if_bad_sheet_by_index(index)
    return self.book.get_sheet_by_index(index)