from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
@property
def header_levels(self) -> int:
    nlevels = self.column_levels
    if self.fmt.has_index_names and self.fmt.show_index_names:
        nlevels += 1
    return nlevels