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
def build_BinOp(ctx, expr):
    lhs = build_expr(ctx, expr.left)
    rhs = build_expr(ctx, expr.right)
    op = type(expr.op)
    if op == ast.Div and (not ctx.uses_true_division):
        err_range = ctx.make_raw_range(lhs.range().end, rhs.range().start)
        raise FrontendError(err_range, 'Division of ints in TorchScript uses Python 3 true division semantics. Please put `from __future__ import division` at the top of your file')
    op_token = ExprBuilder.binop_map.get(op)
    if op_token is None:
        err_range = ctx.make_raw_range(lhs.range().end, rhs.range().start)
        raise NotSupportedError(err_range, 'unsupported binary operator: ' + op.__name__)
    return BinOp(op_token, lhs, rhs)