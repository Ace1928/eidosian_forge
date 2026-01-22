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
def derive(self, blocks, arg_count=None, arg_names=None, force_non_generator=False):
    """
        Derive a new function IR from this one, using the given blocks,
        and possibly modifying the argument count and generator flag.

        Post-processing will have to be run again on the new IR.
        """
    firstblock = blocks[min(blocks)]
    new_ir = copy.copy(self)
    new_ir.blocks = blocks
    new_ir.loc = firstblock.loc
    if force_non_generator:
        new_ir.is_generator = False
    if arg_count is not None:
        new_ir.arg_count = arg_count
    if arg_names is not None:
        new_ir.arg_names = arg_names
    new_ir._reset_analysis_variables()
    new_ir.func_id = new_ir.func_id.derive()
    return new_ir