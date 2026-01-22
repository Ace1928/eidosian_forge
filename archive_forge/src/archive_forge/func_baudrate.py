from __future__ import absolute_import
import io
import time
@baudrate.setter
def baudrate(self, baudrate):
    """        Change baud rate. It raises a ValueError if the port is open and the
        baud rate is not possible. If the port is closed, then the value is
        accepted and the exception is raised when the port is opened.
        """
    try:
        b = int(baudrate)
    except TypeError:
        raise ValueError('Not a valid baudrate: {!r}'.format(baudrate))
    else:
        if b < 0:
            raise ValueError('Not a valid baudrate: {!r}'.format(baudrate))
        self._baudrate = b
        if self.is_open:
            self._reconfigure_port()