from numpy.compat import unicode
from numpy.core._string_helpers import english_lower
from numpy.core.multiarray import typeinfo, dtype
from numpy.core._dtype import _kind_name
def bitname(obj):
    """Return a bit-width name for a given type object"""
    bits = _bits_of(obj)
    dt = dtype(obj)
    char = dt.kind
    base = _kind_name(dt)
    if base == 'object':
        bits = 0
    if bits != 0:
        char = '%s%d' % (char, bits // 8)
    return (base, bits, char)