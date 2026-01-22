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
def create_inplace(cls, kernel, inputs: Union[TensorBox, List[TensorBox]], *args, **kwargs) -> None:
    with V.graph.fake_mode:
        example_output, tensor_args, non_tensor_args, unflatten_args = cls.process_kernel(kernel, inputs, *args, **kwargs)
    for tensor_arg in tensor_args:
        tensor_arg.realize()
    packed = cls(NoneLayout(tensor_args[0].get_device()), kernel, tensor_args, non_tensor_args, unflatten_args)
    pytree.tree_map(lambda x: MutationOutput(x.layout, x, packed), inputs)