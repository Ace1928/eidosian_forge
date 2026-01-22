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
def render_range_for_constraint_violation(self, source, c):
    if isinstance(c, StrictMinMaxConstraint):
        lower, upper = (c.vr.lower, c.vr.upper)
        default = self._default_value_range()
        if lower <= default.lower:
            lower = None
        if upper >= default.upper:
            upper = None
        c_render = f'{self.debug_name(source)} = {source.name()} in the specified range'
        if lower is not None and upper is not None:
            c_render += f' {lower} <= {self.debug_name(source)} <= {upper}'
        elif lower is None and upper is not None:
            c_render += f' {self.debug_name(source)} <= {upper}'
        elif lower is not None and upper is None:
            c_render += f' {lower} <= {self.debug_name(source)}'
        return c_render
    return c.render(source)