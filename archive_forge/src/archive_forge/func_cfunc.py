import sys
import warnings
import inspect
import logging
from numba.core.errors import DeprecationError, NumbaDeprecationWarning
from numba.stencils.stencil import stencil
from numba.core import config, extending, sigutils, registry
def cfunc(sig, locals={}, cache=False, pipeline_class=None, **options):
    """
    This decorator is used to compile a Python function into a C callback
    usable with foreign C libraries.

    Usage::
        @cfunc("float64(float64, float64)", nopython=True, cache=True)
        def add(a, b):
            return a + b

    """
    sig = sigutils.normalize_signature(sig)

    def wrapper(func):
        from numba.core.ccallback import CFunc
        additional_args = {}
        if pipeline_class is not None:
            additional_args['pipeline_class'] = pipeline_class
        res = CFunc(func, sig, locals=locals, options=options, **additional_args)
        if cache:
            res.enable_caching()
        res.compile()
        return res
    return wrapper