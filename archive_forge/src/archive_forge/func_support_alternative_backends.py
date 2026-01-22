import os
import sys
import functools
import numpy as np
from scipy._lib._array_api import array_namespace, is_cupy, is_torch, is_numpy
from . import _ufuncs
from ._ufuncs import (
def support_alternative_backends(f_name, n_array_args):
    func = getattr(_ufuncs, f_name)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        xp = array_namespace(*args[:n_array_args])
        f = get_array_special_func(f_name, xp, n_array_args)
        return f(*args, **kwargs)
    return wrapped