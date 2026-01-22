import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def force_the_name(self, forcename):
    StructOrUnionOrEnum.force_the_name(self, forcename)
    if self.forcename is None:
        name = self.get_official_name()
        self.forcename = '$' + name.replace(' ', '_')