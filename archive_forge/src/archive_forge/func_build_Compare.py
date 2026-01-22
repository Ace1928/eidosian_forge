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
@staticmethod
def build_Compare(ctx, expr):
    operands = [build_expr(ctx, e) for e in [expr.left] + list(expr.comparators)]
    result = None
    for lhs, op_, rhs in zip(operands, expr.ops, operands[1:]):
        op = type(op_)
        op_token = ExprBuilder.cmpop_map.get(op)
        r = ctx.make_raw_range(lhs.range().end, rhs.range().start)
        if op_token is None:
            raise NotSupportedError(r, 'unsupported comparison operator: ' + op.__name__)
        if op == ast.NotIn:
            in_expr = BinOp('in', lhs, rhs)
            cmp_expr = UnaryOp(r, 'not', in_expr)
        else:
            cmp_expr = BinOp(op_token, lhs, rhs)
        if result is None:
            result = cmp_expr
        else:
            result = BinOp('and', result, cmp_expr)
    return result