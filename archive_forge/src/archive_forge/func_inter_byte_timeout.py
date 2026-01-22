from __future__ import absolute_import
import io
import time
@inter_byte_timeout.setter
def inter_byte_timeout(self, ic_timeout):
    """Change inter-byte timeout setting."""
    if ic_timeout is not None:
        if ic_timeout < 0:
            raise ValueError('Not a valid timeout: {!r}'.format(ic_timeout))
        try:
            ic_timeout + 1
        except TypeError:
            raise ValueError('Not a valid timeout: {!r}'.format(ic_timeout))
    self._inter_byte_timeout = ic_timeout
    if self.is_open:
        self._reconfigure_port()