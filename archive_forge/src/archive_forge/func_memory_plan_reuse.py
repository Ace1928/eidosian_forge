import collections
import contextlib
import dataclasses
import functools
import inspect
import os
import re
from itertools import chain, count
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import sympy
from sympy import Expr
import torch
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.fx.node import _get_qualified_name
from torch.utils._sympy.singleton_int import SingletonInt
from .. import codecache, config, ir
from ..codecache import CudaKernelParamCache
from ..ir import ComputedBuffer, InputBuffer, ReinterpretView
from ..triton_heuristics import grid as default_grid
from ..utils import (
from ..virtualized import V
from .common import CodeGen, DeferredLine, IndentedBuffer, PythonPrinter
from .triton_utils import config_of, signature_to_meta
def memory_plan_reuse(self):
    out_names = V.graph.get_output_names()
    while self.lines and isinstance(self.lines[-1], MemoryPlanningLine) and (self.lines[-1].node.name not in out_names):
        self.lines.pop()
    planning_state = MemoryPlanningState()
    for i in range(len(self.lines)):
        if isinstance(self.lines[i], MemoryPlanningLine):
            self.lines[i] = self.lines[i].plan(planning_state)