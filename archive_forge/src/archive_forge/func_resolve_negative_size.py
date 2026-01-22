import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import (
from unittest.mock import patch
import sympy
from sympy import Expr, Integer
import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._prims_common import (
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .dependencies import (
from .utils import (
from .virtualized import ops, V
@staticmethod
def resolve_negative_size(old_size, new_size):
    new_size = [V.graph.sizevars.simplify(x) for x in new_size]
    old_size = [V.graph.sizevars.simplify(x) for x in old_size]
    new_size = list(new_size)
    for i in range(len(new_size)):
        if new_size[i] == -1:
            new_size[i] = sympy.Integer(1)
            new_size[i] = CleanDiv(sympy_product(old_size), sympy_product(new_size))
            break
    V.graph.sizevars.guard_equals(sympy_product(old_size), sympy_product(new_size))
    return (old_size, new_size)