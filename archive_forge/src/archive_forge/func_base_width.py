import copy
import io
from collections import (
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
from . import (
@classmethod
def base_width(cls, num_cols: int, *, column_borders: bool=True, padding: int=1) -> int:
    """
        Utility method to calculate the display width required for a table before data is added to it.
        This is useful when determining how wide to make your columns to have a table be a specific width.

        :param num_cols: how many columns the table will have
        :param column_borders: if True, borders between columns will be included in the calculation (Defaults to True)
        :param padding: number of spaces between text and left/right borders of cell
        :return: base width
        :raises: ValueError if num_cols is less than 1
        """
    if num_cols < 1:
        raise ValueError('Column count cannot be less than 1')
    data_str = SPACE
    data_width = ansi.style_aware_wcswidth(data_str) * num_cols
    tbl = cls([Column(data_str)] * num_cols, column_borders=column_borders, padding=padding)
    data_row = tbl.generate_data_row([data_str] * num_cols)
    return ansi.style_aware_wcswidth(data_row) - data_width