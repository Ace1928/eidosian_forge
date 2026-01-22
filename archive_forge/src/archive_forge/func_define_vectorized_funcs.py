from __future__ import division
from builtins import next
from builtins import zip
from builtins import range
import sys
import inspect
import numpy
from numpy.core import numeric
import uncertainties.umath_core as umath_core
import uncertainties.core as uncert_core
from uncertainties.core import deprecation
def define_vectorized_funcs():
    """
    Defines vectorized versions of functions from uncertainties.umath_core.

    Some functions have their name translated, so as to follow NumPy's
    convention (example: math.acos -> numpy.arccos).
    """
    this_module = sys.modules[__name__]
    func_name_translations = dict([(f_name, 'arc' + f_name[1:]) for f_name in ['acos', 'acosh', 'asin', 'atan', 'atan2', 'atanh']])
    new_func_names = [func_name_translations.get(function_name, function_name) for function_name in umath_core.many_scalars_to_scalar_funcs]
    for function_name, unumpy_name in zip(umath_core.many_scalars_to_scalar_funcs, new_func_names):
        func = getattr(umath_core, function_name)
        otypes = {} if function_name in umath_core.locally_cst_funcs else {'otypes': [object]}
        setattr(this_module, unumpy_name, numpy.vectorize(func, doc='Vectorized version of umath.%s.\n\nOriginal documentation:\n%s' % (function_name, func.__doc__), **otypes))
        __all__.append(unumpy_name)