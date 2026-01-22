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
def build_While(ctx, stmt):
    if stmt.orelse:
        raise NotSupportedError(None, "else branches of while loops aren't supported")
    r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len('while'))
    return While(r, build_expr(ctx, stmt.test), build_stmts(ctx, stmt.body))