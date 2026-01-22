from __future__ import annotations
import numbers
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import register_extension_dtype
from pandas.core.dtypes.missing import isna
from pandas.core import ops
from pandas.core.array_algos import masked_accumulations
from pandas.core.arrays.masked import (
def map_string(s) -> bool:
    if s in true_values_union:
        return True
    elif s in false_values_union:
        return False
    else:
        raise ValueError(f'{s} cannot be cast to bool')