from __future__ import annotations
import os
import abc
from lazyops.libs.pooler import ThreadPooler
from typing import Optional, List, Dict, Any, Union, Type, Tuple, Generator, TYPE_CHECKING
from lazyops.imports._gspread import resolve_gspread
import gspread
def resize_height(self, height: int, wks: Optional[gspread.worksheet.Worksheet]=None) -> None:
    """
        Resize the height of the worksheet
        """
    if not wks:
        wks = self.wks
    sheet_id = wks._properties['sheetId']
    body = {'requests': [{'updateDimensionProperties': {'range': {'sheetId': sheet_id, 'dimension': 'ROWS', 'startIndex': 0, 'endIndex': wks.row_count}, 'properties': {'pixelSize': height}, 'fields': 'pixelSize'}}]}
    wks.batch_update(body)