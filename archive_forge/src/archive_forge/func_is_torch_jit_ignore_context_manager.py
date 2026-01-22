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
def is_torch_jit_ignore_context_manager(stmt):
    if isinstance(stmt.items[0].context_expr, ast.Call):
        function = stmt.items[0].context_expr.func
        if isinstance(function, ast.Attribute):
            attr_name = function.attr
            attr_value = function.value
            if attr_name == '_IgnoreContextManager' and isinstance(attr_value, ast.Attribute):
                if attr_value.attr == 'jit' and isinstance(attr_value.value, ast.Name):
                    if attr_value.value.id == 'torch':
                        return True
    return False