import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def set_shape_setitem(self, obj, shape):
    """remember shapes of SetItem IR nodes.
        """
    assert isinstance(obj, (ir.StaticSetItem, ir.SetItem))
    self.ext_shapes[obj] = shape