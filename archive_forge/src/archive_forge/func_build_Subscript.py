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
def build_Subscript(ctx, expr):

    def build_SliceExpr(ctx, base, slice_expr):
        lower = build_expr(ctx, slice_expr.lower) if slice_expr.lower is not None else None
        upper = build_expr(ctx, slice_expr.upper) if slice_expr.upper is not None else None
        step = build_expr(ctx, slice_expr.step) if slice_expr.step is not None else None
        return SliceExpr(base.range(), lower, upper, step)

    def build_Index(ctx, base, index_expr):
        if isinstance(index_expr.value, ast.Tuple):
            raise NotSupportedError(base.range(), 'slicing multiple dimensions with tuples not supported yet')
        return build_expr(ctx, index_expr.value)

    def build_ExtSlice(ctx, base, extslice):
        sub_exprs = []
        for expr in extslice.dims:
            sub_type = type(expr)
            if sub_type is ast.Index:
                sub_exprs.append(build_Index(ctx, base, expr))
            elif sub_type is ast.Slice:
                sub_exprs.append(build_SliceExpr(ctx, base, expr))
            elif sub_type is ast.Ellipsis:
                sub_exprs.append(Dots(base.range()))
            else:
                raise NotSupportedError(base.range(), f'slicing multiple dimensions with {sub_type} not supported')
        return sub_exprs
    base = build_expr(ctx, expr.value)
    sub_type = type(expr.slice)
    if sub_type is ast.Index:
        if isinstance(expr.slice.value, ast.Tuple):
            indices = [build_expr(ctx, index_expr) for index_expr in expr.slice.value.elts]
            if not indices:
                r = ctx.make_range(expr.lineno, expr.slice.value.col_offset, expr.slice.value.col_offset + 2)
                tup = TupleLiteral(r, [])
                indices.append(tup)
            return Subscript(base, indices)
        else:
            return Subscript(base, [build_expr(ctx, expr.slice.value)])
    elif sub_type is ast.Slice:
        return Subscript(base, [build_SliceExpr(ctx, base, expr.slice)])
    elif sub_type is ast.ExtSlice:
        return Subscript(base, build_ExtSlice(ctx, base, expr.slice))
    elif sys.version_info >= (3, 9):
        if sub_type is ast.Tuple:
            indices = []
            for index_expr in expr.slice.elts:
                if isinstance(index_expr, ast.Slice):
                    indices.append(build_SliceExpr(ctx, base, index_expr))
                else:
                    indices.append(build_expr(ctx, index_expr))
            if not indices:
                r = ctx.make_range(expr.lineno, expr.slice.col_offset, expr.slice.col_offset + 2)
                tup = TupleLiteral(r, [])
                indices.append(tup)
            return Subscript(base, indices)
        return Subscript(base, [build_expr(ctx, expr.slice)])
    else:
        raise NotSupportedError(base.range(), 'ellipsis is not supported')