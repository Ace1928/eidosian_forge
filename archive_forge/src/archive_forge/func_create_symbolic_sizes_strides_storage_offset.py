import builtins
import collections
import functools
import inspect
import itertools
import logging
import math
import operator
import re
import sys
import threading
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, cast, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, Iterable
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx.experimental import _config as config
from torch.fx.experimental.recording import (
from torch.fx.experimental.sym_node import SymNode, SymTypes
from torch import SymBool, SymFloat, SymInt
from torch._guards import ShapeGuard, Source, TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import FloorDiv, Mod, IsNonOverlappingAndDenseIndicator
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.value_ranges import bound_sympy, SymPyValueRangeAnalysis, ValueRanges, ValueRangeError
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._traceback import format_frame, CapturedTraceback
from torch._utils_internal import signpost_event
from torch._logging import LazyString
import sympy
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
def create_symbolic_sizes_strides_storage_offset(self, ex: torch.Tensor, source: Source, *, symbolic_context: Optional[SymbolicContext]=None):
    """
        Returns a list of symbolic sizes and strides for the given tensor.
        We try our best to express stride in terms of the sizes, so as to not
        introduce new symbolic variables.
        """
    assert not ex.is_nested

    def maybe_specialize_sym_int_with_hint(maybe_sym) -> int:
        assert isinstance(maybe_sym, (int, torch.SymInt))
        if is_symbolic(maybe_sym):
            assert maybe_sym.node.shape_env is not self, 'expect the symbol is created from an shape env other than current one.'
            return maybe_sym.node.require_hint()
        return maybe_sym
    ex_size = tuple((maybe_specialize_sym_int_with_hint(sz) for sz in ex.size()))
    ex_stride = tuple((maybe_specialize_sym_int_with_hint(sd) for sd in ex.stride()))
    ex_storage_offset = maybe_specialize_sym_int_with_hint(ex.storage_offset())
    return self._create_symbolic_sizes_strides_storage_offset(ex_size, ex_stride, ex_storage_offset, [_is_dim_dynamic(ex, i) for i in range(ex.dim())], source, symbolic_context=symbolic_context)