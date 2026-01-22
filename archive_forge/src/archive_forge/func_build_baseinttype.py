import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def build_baseinttype(self, ffi, finishlist):
    if self.baseinttype is not None:
        return self.baseinttype.get_cached_btype(ffi, finishlist)
    if self.enumvalues:
        smallest_value = min(self.enumvalues)
        largest_value = max(self.enumvalues)
    else:
        import warnings
        try:
            __warningregistry__.clear()
        except NameError:
            pass
        warnings.warn("%r has no values explicitly defined; guessing that it is equivalent to 'unsigned int'" % self._get_c_name())
        smallest_value = largest_value = 0
    if smallest_value < 0:
        sign = 1
        candidate1 = PrimitiveType('int')
        candidate2 = PrimitiveType('long')
    else:
        sign = 0
        candidate1 = PrimitiveType('unsigned int')
        candidate2 = PrimitiveType('unsigned long')
    btype1 = candidate1.get_cached_btype(ffi, finishlist)
    btype2 = candidate2.get_cached_btype(ffi, finishlist)
    size1 = ffi.sizeof(btype1)
    size2 = ffi.sizeof(btype2)
    if smallest_value >= -1 << 8 * size1 - 1 and largest_value < 1 << 8 * size1 - sign:
        return btype1
    if smallest_value >= -1 << 8 * size2 - 1 and largest_value < 1 << 8 * size2 - sign:
        return btype2
    raise CDefError("%s values don't all fit into either 'long' or 'unsigned long'" % self._get_c_name())