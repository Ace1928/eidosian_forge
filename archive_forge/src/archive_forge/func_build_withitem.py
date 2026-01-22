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
def build_withitem(ctx, item):
    lineno = item.context_expr.lineno
    start = item.context_expr.col_offset
    end = start + len(pretty_node_names[ast.With])
    op_vars = item.optional_vars
    r = ctx.make_range(lineno, start, end)
    return WithItem(r, build_expr(ctx, item.context_expr), build_expr(ctx, op_vars) if op_vars else None)