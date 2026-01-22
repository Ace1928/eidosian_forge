import copy
import io
from collections import (
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
from . import (
def total_width(self) -> int:
    """Calculate the total display width of this table"""
    base_width = self.base_width(len(self.cols), column_borders=self.column_borders, padding=self.padding)
    data_width = sum((col.width for col in self.cols))
    return base_width + data_width