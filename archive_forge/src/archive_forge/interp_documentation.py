import functools
from typing import Any, Dict, Union
import sympy
from sympy.logic.boolalg import Boolean as SympyBoolean, BooleanAtom
import torch
from .functions import (

This is a simple interpreter for Sympy expressions that dispatches to
classes following the torch._inductor.virtualized calling convention.
For directness, the interpreter takes the handler directly rather than
consulting the TLS.  It does not use most of the methods on the full
handler; only those with corresponding Sympy expressions.  To see an example
of a full handler, see torch.utils._sympy.value_ranges.ValueRangeAnalysis.
