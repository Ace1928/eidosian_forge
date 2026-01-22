import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
def register_numpy_ufunc(name, register_global=infer_global):
    func = getattr(np, name)

    class typing_class(Numpy_rules_ufunc):
        key = func
    typing_class.__name__ = 'resolve_{0}'.format(name)
    aliases = ('abs', 'bitwise_not', 'divide', 'abs')
    if name not in aliases:
        register_global(func, types.Function(typing_class))