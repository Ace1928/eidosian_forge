import numpy as np
from functools import wraps
from scipy._lib._docscrape import FunctionDoc, Parameter
from scipy._lib._util import _contains_nan, AxisError, _get_nan
import inspect
def is_too_small(samples, *ts_args, axis=-1, **ts_kwargs):
    for sample in samples:
        if sample.shape[axis] <= too_small:
            return True
    return False