import ast
import dataclasses
import inspect
import re
import string
import sys
from collections import namedtuple
from textwrap import dedent
from typing import List, Tuple  # noqa: F401
import torch
import torch.jit.annotations
from torch import _jit_internal
from torch._C._jit_tree_views import (
from torch._jit_internal import (  # noqa: F401
from torch._sources import (
from torch.jit._dataclass_impls import DATACLASS_MAGIC_METHODS
from torch.jit._monkeytype_config import get_qualified_name, monkeytype_trace
def get_jit_def(fn, def_name, self_name=None, is_classmethod=False):
    """
    Build a JIT AST (TreeView) from the given function.

    Args:
        fn: A function object to compile or a pre-parsed ParsedDef object
        def_name: The name to give to the resulting AST object. This is not
            always the same as `fn.__name__`, for example:
                def _forward(self):
                    ...
                forward = _forward
            In this case, the `__name__` attribute of the function object is "_forward",
            but we want the result AST to have the name "forward".
        self_name: If this function is a method, what the type name of `self` is.
    """
    parsed_def = parse_def(fn) if not isinstance(fn, _ParsedDef) else fn
    type_line = torch.jit.annotations.get_type_line(parsed_def.source)
    fn_def = parsed_def.ast.body[0]
    if is_classmethod:
        arg_name = fn_def.args.args[0].arg
        assign_stmt = ast.parse(f'{arg_name} = {self_name}').body[0]
        fn_def.body.insert(0, assign_stmt)
    if should_drop(fn):
        unused_fn_def = ast.parse('def unused_fn(self: Any):\n\traise RuntimeError("Cannot call @unused methods")')
        if len(unused_fn_def.body) != 1 or not isinstance(unused_fn_def.body[0], ast.FunctionDef):
            raise RuntimeError(f'Expected a single top-level function: {parsed_def.filename}:{parsed_def.file_lineno}')
        unused_def = unused_fn_def.body[0]
        fn_def.body = unused_def.body
        fn_def.args.kwarg = fn_def.args.vararg = None
        for arg in fn_def.args.args + fn_def.args.kwonlyargs:
            arg.annotation = unused_def.args.args[0].annotation
        if _is_drop_fn(fn):
            fn_def.returns = None
            fn_def.type_comment = None
    type_trace_db = torch.jit._script._get_type_trace_db()
    pdt_arg_types = None
    if monkeytype_trace and (not isinstance(fn, _ParsedDef)):
        qualname = get_qualified_name(fn)
        pdt_arg_types = type_trace_db.get_args_types(qualname)
    return build_def(parsed_def.ctx, fn_def, type_line, def_name, self_name=self_name, pdt_arg_types=pdt_arg_types)