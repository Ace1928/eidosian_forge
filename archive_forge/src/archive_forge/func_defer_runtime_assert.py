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
@record_shapeenv_event(save_tracked_fakes=True)
def defer_runtime_assert(self, orig_expr: 'sympy.Expr', msg, fx_node=None):
    expr = orig_expr
    static_expr = self._maybe_evaluate_static(expr)
    if static_expr is not None:
        self.log.debug('runtime_assert %s == %s [statically known]', orig_expr, static_expr)
        return static_expr
    new_expr = self._maybe_evaluate_static(expr, unbacked_only=True)
    if new_expr.free_symbols <= self.var_to_val.keys():
        return self.evaluate_expr(new_expr, fx_node=fx_node)
    if self._translation_validation_enabled and fx_node is not None and (not self._suppress_guards_tls()):
        node, fresh = self.create_fx_call_function(torch._assert, (fx_node,))
        assert node is not None
        if fresh:
            self.add_fx_node_metadata(node)
    self._check_frozen(expr, sympy.true)
    if isinstance(expr, sympy.Eq):
        self._maybe_guard_eq(expr, True)
    if not self._suppress_guards_tls():
        expr = canonicalize_bool_expr(expr)
        stack = CapturedTraceback.extract(skip=1)
        ra = RuntimeAssert(expr, msg, stack)
        cands = sorted([s for s in expr.free_symbols if s.name.startswith('i')], key=lambda s: int(s.name[1:]))
        self.deferred_runtime_asserts.setdefault(cands[-1], []).append(ra)
        self.num_deferred_runtime_asserts += 1
        self._update_version_counter()
        self._log_guard('runtime_assert', expr)
    else:
        self.log.debug('runtime_assert %s [guard suppressed]', expr)
    return True