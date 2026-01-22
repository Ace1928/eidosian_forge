from ..push import PushResult
from .errors import GitSmartRemoteNotSupported
@property
def old_revno(self):
    return self._lookup_revno(self.old_revid)