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
def issue_guard(guard: ShapeGuard) -> None:
    expr = self.simplify(guard.expr)
    if expr in issued:
        return
    issued.add(expr)
    try:
        is_trivial = False
        if any((is_dim(source) for s in expr.free_symbols for source in symbol_to_source[s])):
            is_trivial = self.dim_constraints.add(expr)
        guard_expr = ShapeGuardPrinter(symbol_to_source, source_ref, self.var_to_sources).doprint(expr)
        exprs.append(guard_expr)
        self._add_target_expr(expr)
        if not is_trivial and len(expr.free_symbols) == 1:
            symbol = next(iter(expr.free_symbols))
            source = symbol_to_source[symbol][0]
            constraints = symbol_to_constraints[symbol]
            for c in constraints:
                if isinstance(c, StrictMinMaxConstraint):
                    var_with_range = self.render_range_for_constraint_violation(source, c)
                    msg = f'Not all values of {var_with_range} satisfy the generated guard {guard_expr}.'
                    record_constraint_violation(c.warn_only, self.debug_name(source), msg)
                elif isinstance(c, RelaxedUnspecConstraint):
                    pass
                else:
                    raise AssertionError(f'unrecognized constraint {c}')
    except Exception:
        self.log.warning('Failing guard allocated at: \n%s', ''.join(guard.stack.format()))
        raise