import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def is_char_type(self):
    return self.ALL_PRIMITIVE_TYPES[self.name] == 'c'