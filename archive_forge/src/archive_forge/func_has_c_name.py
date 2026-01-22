import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def has_c_name(self):
    return '$' not in self._get_c_name()