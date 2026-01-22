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
def build_map(cls, items, size, literal_value, value_indexes, loc):
    assert isinstance(loc, Loc)
    op = 'build_map'
    return cls(op=op, loc=loc, items=items, size=size, literal_value=literal_value, value_indexes=value_indexes)