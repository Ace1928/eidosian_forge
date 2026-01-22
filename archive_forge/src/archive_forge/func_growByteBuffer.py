from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def growByteBuffer(self):
    """Doubles the size of the byteslice, and copies the old data towards
           the end of the new buffer (since we build the buffer backwards)."""
    if len(self.Bytes) == Builder.MAX_BUFFER_SIZE:
        msg = 'flatbuffers: cannot grow buffer beyond 2 gigabytes'
        raise BuilderSizeError(msg)
    newSize = min(len(self.Bytes) * 2, Builder.MAX_BUFFER_SIZE)
    if newSize == 0:
        newSize = 1
    bytes2 = bytearray(newSize)
    bytes2[newSize - len(self.Bytes):] = self.Bytes
    self.Bytes = bytes2