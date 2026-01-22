from collections import defaultdict
import copy
import itertools
import os
import linecache
import pprint
import re
import sys
import operator
from types import FunctionType, BuiltinFunctionType
from functools import total_ordering
from io import StringIO
from numba.core import errors, config
from numba.core.utils import (BINOPS_TO_OPERATORS, INPLACE_BINOPS_TO_OPERATORS,
from numba.core.errors import (NotDefinedError, RedefinedError,
from numba.core import consts
@classmethod
def binop(cls, fn, lhs, rhs, loc):
    assert isinstance(fn, BuiltinFunctionType)
    assert isinstance(lhs, Var)
    assert isinstance(rhs, Var)
    assert isinstance(loc, Loc)
    op = 'binop'
    return cls(op=op, loc=loc, fn=fn, lhs=lhs, rhs=rhs, static_lhs=UNDEFINED, static_rhs=UNDEFINED)