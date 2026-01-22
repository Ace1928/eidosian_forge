import dataclasses
import itertools
import sympy
from sympy.logic.boolalg import BooleanAtom, Boolean as SympyBoolean
import operator
import math
import logging
import torch
from typing import Union, Dict, Optional, SupportsFloat
from torch._prims_common import dtype_to_type
from .interp import sympy_interp
def fn_(x, y):
    if x.is_Integer and y.is_Integer:
        result_type = sympy.Integer
    elif x.is_rational and y.is_rational:
        result_type = sympy.Rational
    else:
        assert x.is_real or not x.is_finite or y.is_real or (not y.is_finite)
        result_type = sympy.Float
    return fn(result_type(x), result_type(y))