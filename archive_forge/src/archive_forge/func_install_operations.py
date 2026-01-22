import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
@classmethod
def install_operations(cls):
    for op, ufunc_name in cls._op_map.items():
        infer_global(op)(type('NumpyRulesArrayOperator_' + ufunc_name, (cls,), dict(key=op)))