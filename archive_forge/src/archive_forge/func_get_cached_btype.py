import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def get_cached_btype(self, ffi, finishlist, can_delay=False):
    BType = StructOrUnionOrEnum.get_cached_btype(self, ffi, finishlist, can_delay)
    if not can_delay:
        self.finish_backend_type(ffi, finishlist)
    return BType