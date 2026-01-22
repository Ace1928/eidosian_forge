from __future__ import absolute_import
import io
import time
@rtscts.setter
def rtscts(self, rtscts):
    """Change RTS/CTS flow control setting."""
    self._rtscts = rtscts
    if self.is_open:
        self._reconfigure_port()