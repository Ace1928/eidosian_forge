import rpy2.robjects as ro
import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
from rpy2.rinterface import (Sexp,
import numpy  # type: ignore
import warnings
@rpy2py.register(rinterface.CharSexp)
def rpy2py_charvector(obj):
    if obj == rinterface.NA_Character:
        return None
    else:
        return obj