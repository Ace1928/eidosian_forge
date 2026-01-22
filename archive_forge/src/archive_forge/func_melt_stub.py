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
def melt_stub(df, stub: str, i, j, value_vars, sep: str):
    newdf = melt(df, id_vars=i, value_vars=value_vars, value_name=stub.rstrip(sep), var_name=j)
    newdf[j] = newdf[j].str.replace(re.escape(stub + sep), '', regex=True)
    try:
        newdf[j] = to_numeric(newdf[j])
    except (TypeError, ValueError, OverflowError):
        pass
    return newdf.set_index(i + [j])