from ..push import PushResult
from .errors import GitSmartRemoteNotSupported
@property
def new_revno(self):
    return self._lookup_revno(self.new_revid)