import array
from numba.core import types
def decode_pep3118_format(fmt, itemsize):
    """
    Return the Numba type for an item with format string *fmt* and size
    *itemsize* (in bytes).
    """
    if fmt in _pep3118_int_types:
        name = 'int%d' % (itemsize * 8,)
        if fmt.isupper():
            name = 'u' + name
        return types.Integer(name)
    try:
        return _pep3118_scalar_map[fmt.lstrip('=')]
    except KeyError:
        raise ValueError('unsupported PEP 3118 format %r' % (fmt,))