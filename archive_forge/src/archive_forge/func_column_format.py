from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
@column_format.setter
def column_format(self, input_column_format: str | None) -> None:
    """Setter for column format."""
    if input_column_format is None:
        self._column_format = self._get_index_format() + self._get_column_format_based_on_dtypes()
    elif not isinstance(input_column_format, str):
        raise ValueError(f'column_format must be str or unicode, not {type(input_column_format)}')
    else:
        self._column_format = input_column_format