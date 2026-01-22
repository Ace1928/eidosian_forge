import rpy2.robjects as ro
import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
from rpy2.rinterface import (Sexp,
import numpy  # type: ignore
import warnings
def unsignednumpyint_to_rint(intarray):
    """Convert a numpy array of unsigned integers to an R array."""
    if intarray.itemsize >= RINT_SIZE / 8:
        raise ValueError('Cannot convert numpy array of {numpy_type!s} (R integers are signed {RINT_SIZE}-bit integers).'.format(numpy_type=intarray.dtype.type, RINT_SIZE=RINT_SIZE))
    else:
        res = _numpyarray_to_r(intarray, _kinds['i'])
    return res