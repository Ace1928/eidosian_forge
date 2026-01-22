import types as pytypes  # avoid confusion with numba.types
import copy
import ctypes
import numba.core.analysis
from numba.core import types, typing, errors, ir, rewrites, config, ir_utils
from numba.parfors.parfor import internal_prange
from numba.core.ir_utils import (
from numba.core.analysis import (
from numba.core import postproc
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty_inferred
import numpy as np
import operator
import numba.misc.special
def inline_ir(self, caller_ir, block, i, callee_ir, callee_freevars, arg_typs=None):
    """ Inlines the callee_ir in the caller_ir at statement index i of block
        `block`, callee_freevars are the free variables for the callee_ir. If
        the callee_ir is derived from a function `func` then this is
        `func.__code__.co_freevars`. If `arg_typs` is given and the InlineWorker
        instance was initialized with a typemap and calltypes then they will be
        appropriately updated based on the arg_typs.
        """

    def copy_ir(the_ir):
        kernel_copy = the_ir.copy()
        kernel_copy.blocks = {}
        for block_label, block in the_ir.blocks.items():
            new_block = copy.deepcopy(the_ir.blocks[block_label])
            kernel_copy.blocks[block_label] = new_block
        return kernel_copy
    callee_ir = copy_ir(callee_ir)
    if self.validator is not None:
        self.validator(callee_ir)
    callee_ir_original = copy_ir(callee_ir)
    scope = block.scope
    instr = block.body[i]
    call_expr = instr.value
    callee_blocks = callee_ir.blocks
    max_label = max(ir_utils._the_max_label.next(), max(caller_ir.blocks.keys()))
    callee_blocks = add_offset_to_labels(callee_blocks, max_label + 1)
    callee_blocks = simplify_CFG(callee_blocks)
    callee_ir.blocks = callee_blocks
    min_label = min(callee_blocks.keys())
    max_label = max(callee_blocks.keys())
    ir_utils._the_max_label.update(max_label)
    self.debug_print('After relabel')
    _debug_dump(callee_ir)
    callee_scopes = _get_all_scopes(callee_blocks)
    self.debug_print('callee_scopes = ', callee_scopes)
    assert len(callee_scopes) == 1
    callee_scope = callee_scopes[0]
    var_dict = {}
    for var in tuple(callee_scope.localvars._con.values()):
        if not var.name in callee_freevars:
            inlined_name = _created_inlined_var_name(callee_ir.func_id.unique_name, var.name)
            new_var = scope.redefine(inlined_name, loc=var.loc)
            callee_scope.redefine(inlined_name, loc=var.loc)
            var_dict[var.name] = new_var
    self.debug_print('var_dict = ', var_dict)
    replace_vars(callee_blocks, var_dict)
    self.debug_print('After local var rename')
    _debug_dump(callee_ir)
    callee_func = callee_ir.func_id.func
    args = _get_callee_args(call_expr, callee_func, block.body[i].loc, caller_ir)
    if self._permit_update_type_and_call_maps:
        if arg_typs is None:
            raise TypeError('arg_typs should have a value not None')
        self.update_type_and_call_maps(callee_ir, arg_typs)
        callee_blocks = callee_ir.blocks
    self.debug_print('After arguments rename: ')
    _debug_dump(callee_ir)
    _replace_args_with(callee_blocks, args)
    new_blocks = []
    new_block = ir.Block(scope, block.loc)
    new_block.body = block.body[i + 1:]
    new_label = next_label()
    caller_ir.blocks[new_label] = new_block
    new_blocks.append((new_label, new_block))
    block.body = block.body[:i]
    block.body.append(ir.Jump(min_label, instr.loc))
    topo_order = find_topo_order(callee_blocks)
    _replace_returns(callee_blocks, instr.target, new_label)
    if instr.target.name in caller_ir._definitions and call_expr in caller_ir._definitions[instr.target.name]:
        caller_ir._definitions[instr.target.name].remove(call_expr)
    for label in topo_order:
        block = callee_blocks[label]
        block.scope = scope
        _add_definitions(caller_ir, block)
        caller_ir.blocks[label] = block
        new_blocks.append((label, block))
    self.debug_print('After merge in')
    _debug_dump(caller_ir)
    return (callee_ir_original, callee_blocks, var_dict, new_blocks)