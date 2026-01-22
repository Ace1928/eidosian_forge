import errno
import os
from functools import partial
from threading import Thread
from pyudev._os import pipe, poll
from pyudev._util import eintr_retry_call, ensure_byte_string
from pyudev.device import Device
def receive_device(self):
    """
        Receive a single device from the monitor.

        .. warning::

           You *must* call :meth:`start()` before calling this method.

        The caller must make sure, that there are events available in the
        event queue.  The call blocks, until a device is available.

        If a device was available, return ``(action, device)``.  ``device``
        is the :class:`Device` object describing the device.  ``action`` is
        a string describing the action.  Usual actions are:

        ``'add'``
          A device has been added (e.g. a USB device was plugged in)
        ``'remove'``
          A device has been removed (e.g. a USB device was unplugged)
        ``'change'``
          Something about the device changed (e.g. a device property)
        ``'online'``
          The device is online now
        ``'offline'``
          The device is offline now

        Raise :exc:`~exceptions.EnvironmentError`, if no device could be
        read.

        .. deprecated:: 0.16
           Will be removed in 1.0. Use :meth:`Monitor.poll()` instead.
        """
    import warnings
    warnings.warn('Will be removed in 1.0. Use Monitor.poll() instead.', DeprecationWarning)
    device = self.poll()
    return (device.action, device)