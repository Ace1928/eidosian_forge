import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def pointer_cache(ffi, BType):
    return global_cache('?', ffi, 'new_pointer_type', BType)