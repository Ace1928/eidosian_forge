from __future__ import annotations
from collections import defaultdict
from typing import cast
import numpy as np
from pandas.core.dtypes.generic import (
from pandas.core.indexes.api import MultiIndex
def dataframe_from_int_dict(data, frame_template) -> DataFrame:
    result = DataFrame(data, index=frame_template.index)
    if len(result.columns) > 0:
        result.columns = frame_template.columns[result.columns]
    else:
        result.columns = frame_template.columns.copy()
    return result