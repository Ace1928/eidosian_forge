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
@classmethod
def realize_input(cls, x):
    if x is None:
        return NoneAsConstantBuffer()
    if isinstance(x, (sympy.Expr, sympy.logic.boolalg.Boolean, int)):
        return ShapeAsConstantBuffer(x)
    if isinstance(x, Constant):
        return V.graph.add_tensor_constant(torch.tensor(x.value, dtype=x.get_dtype(), device=x.get_device()))
    if isinstance(x, ConstantBuffer):
        return x
    if isinstance(x, TensorBox):
        return cls.realize_input(x.data)
    if isinstance(x, ReinterpretView):
        return x
    if isinstance(x, BaseView):
        x.realize()
        if is_storage_and_layout(x.unwrap_view()):
            try:
                return cls.convert_to_reinterpret_view(x)
            except NotImplementedError:
                pass
    if isinstance(x, StorageBox):
        x.realize()
        return x
    return cls.copy_input(x)