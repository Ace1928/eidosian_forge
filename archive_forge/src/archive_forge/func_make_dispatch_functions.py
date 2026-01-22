import contextvars
from functools import singledispatch
import os
from typing import Any
from typing import Optional
import typing
import warnings
from rpy2.rinterface_lib import _rinterface_capi
import rpy2.rinterface_lib.sexp
import rpy2.rinterface_lib.conversion
import rpy2.rinterface
@staticmethod
def make_dispatch_functions():
    py2rpy = singledispatch(_py2rpy)
    rpy2py = singledispatch(_rpy2py)
    return (py2rpy, rpy2py)