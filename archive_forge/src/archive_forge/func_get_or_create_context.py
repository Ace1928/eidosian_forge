import functools
import threading
from contextlib import contextmanager
from .driver import driver, USE_NV_BINDING
def get_or_create_context(self, devnum):
    """Returns the primary context and push+create it if needed
        for *devnum*.  If *devnum* is None, use the active CUDA context (must
        be primary) or create a new one with ``devnum=0``.
        """
    if devnum is None:
        attached_ctx = self._get_attached_context()
        if attached_ctx is None:
            return self._get_or_create_context_uncached(devnum)
        else:
            return attached_ctx
    else:
        if USE_NV_BINDING:
            devnum = int(devnum)
        return self._activate_context_for(devnum)