import ast
from typing import Any, Callable, Mapping, Optional, Tuple, Type
import numpy
import numpy.typing as npt
import operator
import cupy
from cupy._logic import ops
from cupy._math import arithmetic
from cupy._logic import comparison
from cupy._binary import elementwise
from cupy import _core
from cupyx.jit import _cuda_types
def get_pyfunc(op_type: Type[ast.AST]) -> Callable[..., Any]:
    return _py_ops[op_type]