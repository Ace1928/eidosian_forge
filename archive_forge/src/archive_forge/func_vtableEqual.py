from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def vtableEqual(a, objectStart, b):
    """vtableEqual compares an unwritten vtable to a written vtable."""
    N.enforce_number(objectStart, N.UOffsetTFlags)
    if len(a) * N.VOffsetTFlags.bytewidth != len(b):
        return False
    for i, elem in enumerate(a):
        x = encode.Get(packer.voffset, b, i * N.VOffsetTFlags.bytewidth)
        if x == 0 and elem == 0:
            pass
        else:
            y = objectStart - elem
            if x != y:
                return False
    return True