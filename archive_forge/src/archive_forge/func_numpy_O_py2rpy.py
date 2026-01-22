import rpy2.robjects as ro
import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
from rpy2.rinterface import (Sexp,
import numpy  # type: ignore
import warnings
def numpy_O_py2rpy(o):
    if all((isinstance(x, str) for x in o)):
        res = StrSexpVector(o)
    elif all((isinstance(x, bytes) for x in o)):
        res = ByteSexpVector(o)
    else:
        res = conversion.get_conversion().py2rpy(list(o))
    return res