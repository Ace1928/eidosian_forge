import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def results_c(full_output, r, method):
    if full_output:
        x, funcalls, iterations, flag = r
        results = RootResults(root=x, iterations=iterations, function_calls=funcalls, flag=flag, method=method)
        return (x, results)
    else:
        return r