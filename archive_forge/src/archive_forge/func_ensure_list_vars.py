from __future__ import annotations
import re
from typing import TYPE_CHECKING
import numpy as np
from pandas.util._decorators import Appender
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.indexes.api import MultiIndex
from pandas.core.reshape.concat import concat
from pandas.core.reshape.util import tile_compat
from pandas.core.shared_docs import _shared_docs
from pandas.core.tools.numeric import to_numeric
def ensure_list_vars(arg_vars, variable: str, columns) -> list:
    if arg_vars is not None:
        if not is_list_like(arg_vars):
            return [arg_vars]
        elif isinstance(columns, MultiIndex) and (not isinstance(arg_vars, list)):
            raise ValueError(f'{variable} must be a list of tuples when columns are a MultiIndex')
        else:
            return list(arg_vars)
    else:
        return []