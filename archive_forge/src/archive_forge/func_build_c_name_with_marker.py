import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def build_c_name_with_marker(self):
    name = self.forcename or '%s %s' % (self.kind, self.name)
    self.c_name_with_marker = name + '&'