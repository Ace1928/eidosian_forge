import numbers
import copy
import types as pytypes
from operator import add
import operator
import numpy as np
import numba.parfors.parfor
from numba.core import types, ir, rewrites, config, ir_utils
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.core.typing import signature
from numba.core import  utils, typing
from numba.core.ir_utils import (get_call_table, mk_unique_var,
from numba.core.errors import NumbaValueError
from numba.core.utils import OPERATORS_TO_BUILTINS
from numba.np import numpy_support
def get_stencil_ir(sf, typingctx, args, scope, loc, input_dict, typemap, calltypes):
    """get typed IR from stencil bytecode
    """
    from numba.core.cpu import CPUContext
    from numba.core.registry import cpu_target
    from numba.core.annotations import type_annotations
    from numba.core.typed_passes import type_inference_stage
    stencil_func_ir = sf.kernel_ir.copy()
    stencil_blocks = copy.deepcopy(stencil_func_ir.blocks)
    stencil_func_ir.blocks = stencil_blocks
    name_var_table = ir_utils.get_name_var_table(stencil_func_ir.blocks)
    if 'out' in name_var_table:
        raise ValueError("Cannot use the reserved word 'out' in stencil kernels.")
    from numba.core.registry import cpu_target
    targetctx = cpu_target.target_context
    tp = DummyPipeline(typingctx, targetctx, args, stencil_func_ir)
    rewrites.rewrite_registry.apply('before-inference', tp.state)
    tp.state.typemap, tp.state.return_type, tp.state.calltypes, _ = type_inference_stage(tp.state.typingctx, tp.state.targetctx, tp.state.func_ir, tp.state.args, None)
    type_annotations.TypeAnnotation(func_ir=tp.state.func_ir, typemap=tp.state.typemap, calltypes=tp.state.calltypes, lifted=(), lifted_from=None, args=tp.state.args, return_type=tp.state.return_type, html_output=config.HTML)
    stencil_blocks = ir_utils.add_offset_to_labels(stencil_blocks, ir_utils.next_label())
    min_label = min(stencil_blocks.keys())
    max_label = max(stencil_blocks.keys())
    ir_utils._the_max_label.update(max_label)
    if config.DEBUG_ARRAY_OPT >= 1:
        print('Initial stencil_blocks')
        ir_utils.dump_blocks(stencil_blocks)
    var_dict = {}
    for v, typ in tp.state.typemap.items():
        new_var = ir.Var(scope, mk_unique_var(v), loc)
        var_dict[v] = new_var
        typemap[new_var.name] = typ
    ir_utils.replace_vars(stencil_blocks, var_dict)
    if config.DEBUG_ARRAY_OPT >= 1:
        print('After replace_vars')
        ir_utils.dump_blocks(stencil_blocks)
    for call, call_typ in tp.state.calltypes.items():
        calltypes[call] = call_typ
    arg_to_arr_dict = {}
    for block in stencil_blocks.values():
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Arg):
                if config.DEBUG_ARRAY_OPT >= 1:
                    print('input_dict', input_dict, stmt.value.index, stmt.value.name, stmt.value.index in input_dict)
                arg_to_arr_dict[stmt.value.name] = input_dict[stmt.value.index].name
                stmt.value = input_dict[stmt.value.index]
    if config.DEBUG_ARRAY_OPT >= 1:
        print('arg_to_arr_dict', arg_to_arr_dict)
        print('After replace arg with arr')
        ir_utils.dump_blocks(stencil_blocks)
    ir_utils.remove_dels(stencil_blocks)
    stencil_func_ir.blocks = stencil_blocks
    return (stencil_func_ir, sf.get_return_type(args)[0], arg_to_arr_dict)