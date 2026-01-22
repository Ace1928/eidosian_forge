import contextlib
import os
import typing
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.sexp as sexp
from rpy2.robjects.robject import RObjectMixin
from rpy2.robjects import conversion
def popitem(self) -> typing.Tuple[str, sexp.Sexp]:
    """ E.popitem() -> (k, v), remove and return some (key, value)
        pair as a 2-tuple; but raise KeyError if E is empty. """
    if len(self) == 0:
        raise KeyError()
    kv = next(self.items())
    del self[kv[0]]
    return kv