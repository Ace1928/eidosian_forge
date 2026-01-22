import inspect
import itertools
from . import _funcs_impl, _reductions_impl
from ._normalizations import normalizer
def is_public_function(f):
    return inspect.isfunction(f) and (not f.__name__.startswith('_'))