from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
def traverse_models(self):
    """
        Recursively list all models involved in this model.
        """
    return [self._dmm[t] for t in self.traverse_types()]