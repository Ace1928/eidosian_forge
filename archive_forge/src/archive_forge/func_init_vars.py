import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def init_vars(self, block):
    """
        Initialize live variables for *block*.
        """
    self._live_vars = set(self.func_ir.get_block_entry_vars(block))