import errno
import os
from functools import partial
from threading import Thread
from pyudev._os import pipe, poll
from pyudev._util import eintr_retry_call, ensure_byte_string
from pyudev.device import Device
def set_receive_buffer_size(self, size):
    """
        Set the receive buffer ``size``.

        ``size`` is the requested buffer size in bytes, as integer.

        .. note::

           The CAP_NET_ADMIN capability must be contained in the effective
           capability set of the caller for this method to succeed.  Otherwise
           :exc:`~exceptions.EnvironmentError` will be raised, with ``errno``
           set to :data:`~errno.EPERM`.  Unprivileged processes typically lack
           this capability.  You can check the capabilities of the current
           process with the python-prctl_ module:

           >>> import prctl
           >>> prctl.cap_effective.net_admin

        Raise :exc:`~exceptions.EnvironmentError`, if the buffer size could not
        bet set.

        .. versionadded:: 0.13

        .. _python-prctl: http://packages.python.org/python-prctl
        """
    self._libudev.udev_monitor_set_receive_buffer_size(self, size)