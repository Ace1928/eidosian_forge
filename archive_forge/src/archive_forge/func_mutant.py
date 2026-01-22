import collections
from functools import wraps
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PVector, pvector
def mutant(fn):
    """
    Convenience decorator to isolate mutation to within the decorated function (with respect
    to the input arguments).

    All arguments to the decorated function will be frozen so that they are guaranteed not to change.
    The return value is also frozen.
    """

    @wraps(fn)
    def inner_f(*args, **kwargs):
        return freeze(fn(*[freeze(e) for e in args], **dict((freeze(item) for item in kwargs.items()))))
    return inner_f