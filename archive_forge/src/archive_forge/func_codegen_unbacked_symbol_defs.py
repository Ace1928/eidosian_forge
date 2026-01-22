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
def codegen_unbacked_symbol_defs(self, wrapper):
    symbols_to_define = self.get_unbacked_symbol_defs()
    for i, s in enumerate(self.get_size()):
        if s in symbols_to_define:
            wrapper.writeline(f'{wrapper.codegen_unbacked_symbol_decl(s)} = {self.get_name()}.size({i}){wrapper.ending}')
            symbols_to_define.remove(s)
    for i, s in enumerate(self.get_stride()):
        if s in symbols_to_define:
            wrapper.writeline(f'{wrapper.codegen_unbacked_symbol_decl(s)} = {self.get_name()}.stride({i}){wrapper.ending}')
            symbols_to_define.remove(s)
    if (s := self.get_offset()) in symbols_to_define:
        wrapper.writeline(f'{wrapper.codegen_unbacked_symbol_decl(s)} = {self.get_name()}.storage_offset(){wrapper.ending}')
        symbols_to_define.remove(s)
    assert not symbols_to_define, f'unbacked symint {s} not written out, check comment above'