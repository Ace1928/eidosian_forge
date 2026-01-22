from __future__ import annotations
import operator
import numpy as np
from xarray.core import dtypes, duck_array_ops
def op_str(name):
    return f'__{name}__'