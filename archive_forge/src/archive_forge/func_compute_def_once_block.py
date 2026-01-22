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
def compute_def_once_block(block, def_once, def_more, getattr_taken, typemap, module_assigns):
    """Effect changes to the set of variables defined once or more than once
       for a single block.
       block - the block to process
       def_once - set of variable names known to be defined exactly once
       def_more - set of variable names known to be defined more than once
       getattr_taken - dict mapping variable name to tuple of object and attribute taken
       module_assigns - dict mapping variable name to the Global that they came from
    """
    assignments = block.find_insts(ir.Assign)
    for one_assign in assignments:
        a_def = one_assign.target.name
        add_to_def_once_sets(a_def, def_once, def_more)
        rhs = one_assign.value
        if isinstance(rhs, ir.Global):
            if isinstance(rhs.value, pytypes.ModuleType):
                module_assigns[a_def] = rhs.value.__name__
        if isinstance(rhs, ir.Expr) and rhs.op == 'getattr' and (rhs.value.name in def_once):
            getattr_taken[a_def] = (rhs.value.name, rhs.attr)
        if isinstance(rhs, ir.Expr) and rhs.op == 'call' and (rhs.func.name in getattr_taken):
            base_obj, base_attr = getattr_taken[rhs.func.name]
            if base_obj in module_assigns:
                base_mod_name = module_assigns[base_obj]
                if not is_const_call(base_mod_name, base_attr):
                    add_to_def_once_sets(base_obj, def_once, def_more)
            else:
                add_to_def_once_sets(base_obj, def_once, def_more)
        if isinstance(rhs, ir.Expr) and rhs.op == 'call':
            for argvar in rhs.args:
                if isinstance(argvar, ir.Var):
                    argvar = argvar.name
                avtype = typemap[argvar]
                if getattr(avtype, 'mutable', False):
                    add_to_def_once_sets(argvar, def_once, def_more)