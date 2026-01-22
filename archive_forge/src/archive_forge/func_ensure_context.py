import functools
import threading
from contextlib import contextmanager
from .driver import driver, USE_NV_BINDING
@contextmanager
def ensure_context(self):
    """Ensure a CUDA context is available inside the context.

        On entrance, queries the CUDA driver for an active CUDA context and
        attaches it in TLS for subsequent calls so they do not need to query
        the CUDA driver again.  On exit, detach the CUDA context from the TLS.

        This will allow us to pickup thirdparty activated CUDA context in
        any top-level Numba CUDA API.
        """
    with driver.get_active_context():
        oldctx = self._get_attached_context()
        newctx = self.get_or_create_context(None)
        self._set_attached_context(newctx)
        try:
            yield
        finally:
            self._set_attached_context(oldctx)