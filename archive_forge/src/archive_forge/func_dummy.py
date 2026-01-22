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
def dummy(cls, op, info, loc):
    """
        A node for a dummy value.

        This node is a place holder for carrying information through to a point
        where it is rewritten into something valid. This node is not handled
        by type inference or lowering. It's presence outside of the interpreter
        renders IR as illegal.
        """
    assert isinstance(loc, Loc)
    assert isinstance(op, str)
    return cls(op=op, info=info, loc=loc)