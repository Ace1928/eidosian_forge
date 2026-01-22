from abc import ABC, abstractmethod
from builtins import bool
from typing import Any, Dict, List, Tuple
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PandasLikeUtils
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
def binary_logical_op(self, col1: Column, col2: Column, op: str) -> Column:
    c1 = self._safe_bool(col1)
    c2 = self._safe_bool(col2)
    if op == 'and':
        s: Any = c1 * c2
        if self.is_series(s):
            s = s.mask((c1 == 0) | (c2 == 0), 0)
        elif (c1 == 0) | (c2 == 0):
            s = 0.0
    elif op == 'or':
        s = c1 + c2
        if self.is_series(s):
            s = s.mask((c1 > 0) | (c2 > 0), 1)
        elif (c1 > 0) | (c2 > 0):
            s = 1.0
    else:
        raise NotImplementedError(f'{op} is not supported')
    return Column(s)