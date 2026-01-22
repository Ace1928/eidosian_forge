from numpy import int8, int16, int32, int64, intp, intc
from numpy import uint8, uint16, uint32, uint64, uintp, uintc
from numpy import float64, float32, complex64, complex128
import numpy
from pythran.typing import List, Dict, Set, Tuple, NDArray, Pointer, Fun
def pytype_to_pretty_type(t):
    """ Python -> docstring type. """
    if isinstance(t, List):
        return '{0} list'.format(pytype_to_pretty_type(t.__args__[0]))
    elif isinstance(t, Set):
        return '{0} set'.format(pytype_to_pretty_type(t.__args__[0]))
    elif isinstance(t, Dict):
        tkey, tvalue = t.__args__
        return '{0}:{1} dict'.format(pytype_to_pretty_type(tkey), pytype_to_pretty_type(tvalue))
    elif isinstance(t, Tuple):
        return '({0})'.format(', '.join((pytype_to_pretty_type(p) for p in t.__args__)))
    elif isinstance(t, NDArray):
        dtype = pytype_to_pretty_type(t.__args__[0])
        ndim = len(t.__args__) - 1
        arr = '{0}[{1}]'.format(dtype, ','.join((':' if s.stop in (-1, None) else str(s.stop) for s in t.__args__[1:])))
        if t.__args__[1].start == -1:
            return '{} order(F)'.format(arr)
        elif any((s.step is not None and s.step < 0 for s in t.__args__[1:])):
            return '{0}[{1}]'.format(dtype, ','.join(['::'] * ndim))
        else:
            return arr
    elif isinstance(t, Pointer):
        dtype = pytype_to_pretty_type(t.__args__[0])
        return '{}*'.format(dtype)
    elif isinstance(t, Fun):
        rtype = pytype_to_pretty_type(t.__args__[-1])
        argtypes = [pytype_to_pretty_type(arg) for arg in t.__args__[:-1]]
        return '{}({})'.format(rtype, ', '.join(argtypes))
    elif t in PYTYPE_TO_CTYPE_TABLE:
        return t.__name__
    else:
        raise NotImplementedError('{0}:{1}'.format(type(t), t))