import copy
import operator
import types as pytypes
import operator
import warnings
from dataclasses import make_dataclass
import llvmlite.ir
import numpy as np
import numba
from numba.parfors import parfor
from numba.core import types, ir, config, compiler, sigutils, cgutils
from numba.core.ir_utils import (
from numba.core.typing import signature
from numba.core import lowering
from numba.parfors.parfor import ensure_parallel_support
from numba.core.errors import (
from numba.parfors.parfor_lowering_utils import ParforLoweringBuilder
def replace_var_with_array_in_block(vars, block, typemap, calltypes):
    new_block = []
    for inst in block.body:
        if isinstance(inst, ir.Assign) and inst.target.name in vars:
            loc = inst.loc
            scope = inst.target.scope
            const_node = ir.Const(0, loc)
            const_var = scope.redefine('$const_ind_0', loc)
            typemap[const_var.name] = types.uintp
            const_assign = ir.Assign(const_node, const_var, loc)
            new_block.append(const_assign)
            val_var = scope.redefine('$val', loc)
            typemap[val_var.name] = typemap[inst.target.name]
            new_block.append(ir.Assign(inst.value, val_var, loc))
            setitem_node = ir.SetItem(inst.target, const_var, val_var, loc)
            calltypes[setitem_node] = signature(types.none, types.npytypes.Array(typemap[inst.target.name], 1, 'C'), types.intp, typemap[inst.target.name])
            new_block.append(setitem_node)
            continue
        elif isinstance(inst, parfor.Parfor):
            replace_var_with_array_internal(vars, {0: inst.init_block}, typemap, calltypes)
            replace_var_with_array_internal(vars, inst.loop_body, typemap, calltypes)
        new_block.append(inst)
    return new_block