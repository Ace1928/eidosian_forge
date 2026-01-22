import copy
import numpy as np
from llvmlite import ir as lir
from numba.core import types, typing, utils, ir, config, ir_utils, registry
from numba.core.typing.templates import (CallableTemplate, signature,
from numba.core.imputils import lower_builtin
from numba.core.extending import register_jitable
from numba.core.errors import NumbaValueError
from numba.misc.special import literal_unroll
import numba
import operator
from numba.np import numpy_support
def copy_ir_with_calltypes(self, ir, calltypes):
    """
        Create a copy of a given IR along with its calltype information.
        We need a copy of the calltypes because copy propagation applied
        to the copied IR will change the calltypes and make subsequent
        uses of the original IR invalid.
        """
    copy_calltypes = {}
    kernel_copy = ir.copy()
    kernel_copy.blocks = {}
    for block_label, block in ir.blocks.items():
        new_block = copy.deepcopy(ir.blocks[block_label])
        new_block.body = []
        for stmt in ir.blocks[block_label].body:
            scopy = copy.deepcopy(stmt)
            new_block.body.append(scopy)
            if stmt in calltypes:
                copy_calltypes[scopy] = calltypes[stmt]
        kernel_copy.blocks[block_label] = new_block
    return (kernel_copy, copy_calltypes)