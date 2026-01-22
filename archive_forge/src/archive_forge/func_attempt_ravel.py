from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import operator
import warnings
from functools import reduce
import numpy as np
from numba.np.ufunc.ufuncbuilder import _BaseUFuncBuilder, parse_identity
from numba.core import types, sigutils
from numba.core.typing import signature
from numba.np.ufunc.sigparse import parse_signature
def attempt_ravel(a):
    if cr.SUPPORT_DEVICE_SLICING:
        raise NotImplementedError
    try:
        return a.ravel()
    except NotImplementedError:
        if not cr.is_device_array(a):
            raise
        else:
            hostary = cr.to_host(a, stream).ravel()
            return cr.to_device(hostary, stream)