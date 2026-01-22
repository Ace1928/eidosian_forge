import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def length_is_unknown(self):
    return isinstance(self.length, str)