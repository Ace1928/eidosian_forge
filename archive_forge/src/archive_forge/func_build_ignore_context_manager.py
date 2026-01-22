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
def build_ignore_context_manager(ctx, stmt):
    InputType = namedtuple('InputType', ['name', 'ann'])
    OutputType = namedtuple('OutputType', ['name', 'ann'])

    def process_ins_outs(args):
        inputs = []
        outputs = []
        for arg in args:
            var_name = arg.arg
            var_ann = arg.value.value
            var_decl_type, var_ann = var_ann.split(':')
            if var_decl_type == 'inp':
                inputs.append(InputType(var_name, var_ann))
            if var_decl_type == 'out':
                outputs.append(OutputType(var_name, var_ann))
        return (inputs, outputs)

    def create_unique_name_ext(ctx, stmt):
        fn = re.sub('[^a-zA-Z0-9_]', '_', ctx.filename)
        return f'{fn}_{stmt.lineno}'

    def build_return_ann_stmt(outputs):
        return_type_ann = ''
        return_statement_str = 'return '
        if len(outputs) == 0:
            return_type_ann += ' -> None'
        if len(outputs) == 1:
            return_type_ann = ' -> ' + outputs[0].ann
            return_statement_str += outputs[0].name
        if len(outputs) > 1:
            return_type_ann = ' -> Tuple'
            return_type_ann += '[' + ', '.join([var.ann for var in outputs]) + ']'
            return_statement_str += ', '.join([var.name for var in outputs])
        return (return_type_ann, return_statement_str)

    def build_args(args):
        return ', '.join([arg.name for arg in args])
    inputs, outputs = process_ins_outs(stmt.items[0].context_expr.keywords)
    ignore_function_name = 'func_ignore_' + create_unique_name_ext(ctx, stmt)
    ignore_function_str = '\ndef ' + ignore_function_name
    ignore_function_str += '(' + ', '.join([var.name + ' :' + var.ann for var in inputs]) + ')'
    return_ann, return_stmt = build_return_ann_stmt(outputs)
    ignore_function_str += return_ann + ': pass'
    ignore_function = ast.parse(ignore_function_str).body[0]
    ignore_function.body = stmt.body
    return_stmt = ast.parse(return_stmt).body[0]
    ignore_function.body.append(return_stmt)
    ignore_func_str = '@torch.jit.ignore\n' + astunparse.unparse(ignore_function)
    ignore_func_str += f'\nglobals()["{ignore_function_name}"] = {ignore_function_name}'
    exec(ignore_func_str)
    assign_str_lhs = build_args(outputs)
    assign_str_rhs = f'torch.jit.frontend.{ignore_function_name}(' + build_args(inputs) + ')'
    if len(outputs) > 0:
        assign_str = assign_str_lhs + ' = ' + assign_str_rhs
    else:
        assign_str = assign_str_rhs
    assign_ast = ast.parse(assign_str).body[0]
    return assign_ast