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
def empty_container_allocator_hoist(inst, dep_on_param, call_table, hoisted, not_hoisted, typemap, stored_arrays):
    if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr) and (inst.value.op == 'call') and (inst.value.func.name in call_table):
        call_list = call_table[inst.value.func.name]
        if call_list == ['empty', np]:
            return _hoist_internal(inst, dep_on_param, call_table, hoisted, not_hoisted, typemap, stored_arrays)
    return False