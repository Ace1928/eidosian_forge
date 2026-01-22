import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def compile_to_numba_ir(mk_func, glbls, typingctx=None, targetctx=None, arg_typs=None, typemap=None, calltypes=None):
    """
    Compile a function or a make_function node to Numba IR.

    Rename variables and
    labels to avoid conflict if inlined somewhere else. Perform type inference
    if typingctx and other typing inputs are available and update typemap and
    calltypes.
    """
    from numba.core import typed_passes
    if hasattr(mk_func, 'code'):
        code = mk_func.code
    elif hasattr(mk_func, '__code__'):
        code = mk_func.__code__
    else:
        raise NotImplementedError('function type not recognized {}'.format(mk_func))
    f_ir = get_ir_of_code(glbls, code)
    remove_dels(f_ir.blocks)
    f_ir.blocks = add_offset_to_labels(f_ir.blocks, _the_max_label.next())
    max_label = max(f_ir.blocks.keys())
    _the_max_label.update(max_label)
    var_table = get_name_var_table(f_ir.blocks)
    new_var_dict = {}
    for name, var in var_table.items():
        new_var_dict[name] = mk_unique_var(name)
    replace_var_names(f_ir.blocks, new_var_dict)
    if typingctx:
        f_typemap, f_return_type, f_calltypes, _ = typed_passes.type_inference_stage(typingctx, targetctx, f_ir, arg_typs, None)
        arg_names = [vname for vname in f_typemap if vname.startswith('arg.')]
        for a in arg_names:
            f_typemap.pop(a)
        typemap.update(f_typemap)
        calltypes.update(f_calltypes)
    return f_ir