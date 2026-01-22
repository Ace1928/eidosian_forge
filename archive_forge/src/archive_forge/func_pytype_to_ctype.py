from numpy import int8, int16, int32, int64, intp, intc
from numpy import uint8, uint16, uint32, uint64, uintp, uintc
from numpy import float64, float32, complex64, complex128
import numpy
from pythran.typing import List, Dict, Set, Tuple, NDArray, Pointer, Fun
def pytype_to_ctype(t):
    """ Python -> pythonic type binding. """
    if isinstance(t, List):
        return 'pythonic::types::list<{0}>'.format(pytype_to_ctype(t.__args__[0]))
    elif isinstance(t, Set):
        return 'pythonic::types::set<{0}>'.format(pytype_to_ctype(t.__args__[0]))
    elif isinstance(t, Dict):
        tkey, tvalue = t.__args__
        return 'pythonic::types::dict<{0},{1}>'.format(pytype_to_ctype(tkey), pytype_to_ctype(tvalue))
    elif isinstance(t, Tuple):
        return 'decltype(pythonic::types::make_tuple({0}))'.format(', '.join(('std::declval<{}>()'.format(pytype_to_ctype(p)) for p in t.__args__)))
    elif isinstance(t, NDArray):
        dtype = pytype_to_ctype(t.__args__[0])
        ndim = len(t.__args__) - 1
        shapes = ','.join(('long' if s.stop == -1 or s.stop is None else 'std::integral_constant<long, {}>'.format(s.stop) for s in t.__args__[1:]))
        pshape = 'pythonic::types::pshape<{0}>'.format(shapes)
        arr = 'pythonic::types::ndarray<{0},{1}>'.format(dtype, pshape)
        if t.__args__[1].start == -1:
            return 'pythonic::types::numpy_texpr<{0}>'.format(arr)
        elif any((s.step is not None and s.step < 0 for s in t.__args__[1:])):
            slices = ', '.join(['pythonic::types::normalized_slice'] * ndim)
            return 'pythonic::types::numpy_gexpr<{0},{1}>'.format(arr, slices)
        else:
            return arr
    elif isinstance(t, Pointer):
        return 'pythonic::types::pointer<{0}>'.format(pytype_to_ctype(t.__args__[0]))
    elif isinstance(t, Fun):
        return 'pythonic::types::cfun<{0}({1})>'.format(pytype_to_ctype(t.__args__[-1]), ', '.join((pytype_to_ctype(arg) for arg in t.__args__[:-1])))
    elif t in PYTYPE_TO_CTYPE_TABLE:
        return PYTYPE_TO_CTYPE_TABLE[t]
    else:
        raise NotImplementedError('{0}:{1}'.format(type(t), t))