import ast
import inspect
import re
import sys
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from .. import language
from .._C.libtriton.triton import ir
from ..language import constexpr, tensor
from ..runtime import JITFunction
from .errors import (CompilationError, CompileTimeAssertionFailure, UnsupportedLanguageConstruct)
def ast_to_ttir(fn, signature, specialization, constants, debug, target):
    if isinstance(signature, str):
        signature = {k: v.strip() for k, v in enumerate(signature.split(','))}
    context = ir.context()
    context.load_triton()
    cst_key = lambda i: fn.arg_names.index(i) if isinstance(i, str) else i
    constants = {cst_key(key): value for key, value in constants.items()}
    gscope = fn.__globals__.copy()
    function_name = '_'.join([fn.__name__, kernel_suffix(signature.values(), specialization)])
    tys = list(signature.values())
    new_constants = {k: True if k in tys and tys[k] == 'i1' else 1 for k in specialization.equal_to_1}
    new_attrs = {k: [('tt.divisibility', 16)] for k in specialization.divisible_by_16}
    for k in specialization.divisible_by_8:
        attr = new_attrs[k] if k in new_attrs else []
        if k in specialization.divisible_by_16:
            attr.append(('tt.max_divisibility', 16))
        else:
            attr.append(('tt.max_divisibility', 8))
        new_attrs[k] = attr
    all_constants = constants.copy()
    all_constants.update(new_constants)
    arg_types = [str_to_ty(v) for k, v in signature.items() if k not in constants]
    file_name, begin_line = _get_fn_file_line(fn)
    prototype = language.function_type([], arg_types)
    generator = CodeGenerator(context, prototype, gscope=gscope, constants=all_constants, function_name=function_name, attributes=new_attrs, is_kernel=True, debug=debug, file_name=file_name, begin_line=begin_line, target=target)
    try:
        generator.visit(fn.parse())
    except CompilationError as e:
        if e.src is None:
            e.set_source_code(fn.src)
        raise
    except Exception as e:
        node = generator.last_node
        if node is None:
            raise
        raise CompilationError(fn.src, node, repr(e)) from e
    ret = generator.module
    ret.context = context
    return ret