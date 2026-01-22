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
def build_BoolOp(ctx, expr):
    if len(expr.values) < 2:
        raise AssertionError('expected at least 2 values in BoolOp, but got ' + str(len(expr.values)))
    sub_exprs = [build_expr(ctx, sub_expr) for sub_expr in expr.values]
    op = type(expr.op)
    op_token = ExprBuilder.boolop_map.get(op)
    if op_token is None:
        err_range = ctx.make_raw_range(sub_exprs[0].range().end, sub_exprs[1].range().start)
        raise NotSupportedError(err_range, 'unsupported boolean operator: ' + op.__name__)
    lhs = sub_exprs[0]
    for rhs in sub_exprs[1:]:
        lhs = BinOp(op_token, lhs, rhs)
    return lhs