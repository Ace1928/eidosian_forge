from __future__ import absolute_import
import io
import time
@dsrdtr.setter
def dsrdtr(self, dsrdtr=None):
    """Change DsrDtr flow control setting."""
    if dsrdtr is None:
        self._dsrdtr = self._rtscts
    else:
        self._dsrdtr = dsrdtr
    if self.is_open:
        self._reconfigure_port()