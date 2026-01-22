import functools
import numbers
import sys
import numpy as np
from . import numerictypes as _nt
from .umath import absolute, isinf, isfinite, isnat
from . import multiarray
from .multiarray import (array, dragon4_positional, dragon4_scientific,
from .fromnumeric import any
from .numeric import concatenate, asarray, errstate
from .numerictypes import (longlong, intc, int_, float_, complex_, bool_,
from .overrides import array_function_dispatch, set_module
import operator
import warnings
import contextlib
def format_array(self, a):
    if np.ndim(a) == 0:
        return self.format_function(a)
    if self.summary_insert and a.shape[0] > 2 * self.edge_items:
        formatted = [self.format_array(a_) for a_ in a[:self.edge_items]] + [self.summary_insert] + [self.format_array(a_) for a_ in a[-self.edge_items:]]
    else:
        formatted = [self.format_array(a_) for a_ in a]
    return '[' + ', '.join(formatted) + ']'