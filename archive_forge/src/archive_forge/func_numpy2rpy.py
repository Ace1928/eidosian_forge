import rpy2.robjects as ro
import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
from rpy2.rinterface import (Sexp,
import numpy  # type: ignore
import warnings
@py2rpy.register(numpy.ndarray)
def numpy2rpy(o):
    """ Augmented conversion function, converting numpy arrays into
    rpy2.rinterface-level R structures. """
    if not o.dtype.isnative:
        raise ValueError('Cannot pass numpy arrays with non-native byte orders at the moment.')
    if o.dtype.kind in _kinds:
        res = _numpyarray_to_r(o, _kinds[o.dtype.kind])
    elif o.dtype.kind == 'u':
        res = unsignednumpyint_to_rint(o)
    elif o.dtype.kind == 'O':
        res = numpy_O_py2rpy(o)
    elif o.dtype.kind == 'V':
        if o.dtype.names is None:
            raise ValueError('Nothing can be done for this numpy array type "%s" at the moment.' % (o.dtype,))
        df_args = []
        cv = conversion.get_conversion()
        for field_name in o.dtype.names:
            df_args.append((field_name, cv.py2rpy(o[field_name])))
        res = ro.baseenv['data.frame'].rcall(tuple(df_args))
    else:
        raise ValueError('Unknown numpy array type "%s".' % str(o.dtype))
    return res