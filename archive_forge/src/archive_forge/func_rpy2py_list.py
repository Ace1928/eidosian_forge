import rpy2.robjects as ro
import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
from rpy2.rinterface import (Sexp,
import numpy  # type: ignore
import warnings
def rpy2py_list(obj: rinterface.ListSexpVector):
    if not isinstance(obj, ro.vectors.ListVector):
        obj = ro.vectors.ListVector(obj)
    res = rlc.OrdDict(obj.items())
    return res