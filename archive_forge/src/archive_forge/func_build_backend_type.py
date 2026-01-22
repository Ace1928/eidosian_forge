import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def build_backend_type(self, ffi, finishlist):
    self.check_not_partial()
    base_btype = self.build_baseinttype(ffi, finishlist)
    return global_cache(self, ffi, 'new_enum_type', self.get_official_name(), self.enumerators, self.enumvalues, base_btype, key=self)