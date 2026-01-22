import itertools
import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.python import (
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.gen import cpp_string, parse_native_yaml, parse_tags_yaml
from torchgen.model import (
from torchgen.utils import FileManager, split_name_params
from torchgen.yaml_utils import YamlLoader
from .gen_trace_type import should_trace
def emit_single_dispatch(ps: PythonSignature, f: NativeFunction, namedtuple_typenames: Dict[str, str], *, symint: bool=True) -> str:
    """
    Emit dispatch code for a single native function.
    """

    @with_native_function
    def go(f: NativeFunction) -> str:
        if isinstance(ps, PythonSignatureDeprecated):
            schema_comment = f'// [deprecated] aten::{ps.deprecated_schema}'
        else:
            schema_comment = f'// aten::{f.func}'
        deprecated = '[deprecated] ' if ps.deprecated else ''
        name = cpp.name(f.func)
        lambda_formals = ', '.join((f'{a.type_str} {a.name}' for a in dispatch_lambda_args(ps, f, symint=symint)))
        lambda_return = dispatch_lambda_return_str(f)
        dispatch_callee = cpp_dispatch_target(f)
        dispatch_args = ', '.join(cpp_dispatch_exprs(f, python_signature=ps))
        parser_outputs = arg_parser_output_exprs(ps, f, symint=symint)
        lambda_arg_exprs = dispatch_lambda_exprs(ps, f, symint=symint)
        inits = '\n'.join(lambda_arg_exprs.inits)
        lambda_args = ', '.join(lambda_arg_exprs.exprs)
        need_set_requires_grad = ps.tensor_options_args and (not has_tensor_options(f) or (ps.method and 'requires_grad' in parser_outputs))
        set_requires_grad = f'.set_requires_grad({parser_outputs['requires_grad'].expr})' if need_set_requires_grad else ''
        if lambda_return == 'void':
            return f'{schema_comment}\n{inits}\nauto dispatch_{name} = []({lambda_formals}) -> {lambda_return} {{\n  pybind11::gil_scoped_release no_gil;\n  {dispatch_callee}({dispatch_args});\n}};\ndispatch_{name}({lambda_args}){set_requires_grad};\nPy_RETURN_NONE;\n'
        else:
            typename = namedtuple_typenames.get(gen_namedtuple_typename_key(f))
            namedtuple_typeref = f'{typename}, ' if typename is not None else ''
            return f'{schema_comment}\n{inits}\nauto dispatch_{name} = []({lambda_formals}) -> {lambda_return} {{\n  pybind11::gil_scoped_release no_gil;\n  return {dispatch_callee}({dispatch_args});\n}};\nreturn wrap({namedtuple_typeref}dispatch_{name}({lambda_args}){set_requires_grad});\n'
    return go(f)