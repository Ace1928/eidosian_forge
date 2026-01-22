from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
def get_data_type(self):
    return ir.global_context.get_identified_type(self.typename + '.data')