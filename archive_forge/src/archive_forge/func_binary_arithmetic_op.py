from abc import ABC, abstractmethod
from builtins import bool
from typing import Any, Dict, List, Tuple
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PandasLikeUtils
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
def binary_arithmetic_op(self, col1: Column, col2: Column, op: str) -> Column:
    if op == '+':
        return Column(col1.native + col2.native)
    if op == '-':
        return Column(col1.native - col2.native)
    if op == '*':
        return Column(col1.native * col2.native)
    if op == '/':
        return Column(col1.native / col2.native)
    raise NotImplementedError(f'{op} is not supported')