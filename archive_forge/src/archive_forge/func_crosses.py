from .base import GEOSBase
from .prototypes import prepared as capi
def crosses(self, other):
    return capi.prepared_crosses(self.ptr, other.ptr)