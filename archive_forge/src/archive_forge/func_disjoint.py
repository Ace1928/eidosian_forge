from .base import GEOSBase
from .prototypes import prepared as capi
def disjoint(self, other):
    return capi.prepared_disjoint(self.ptr, other.ptr)