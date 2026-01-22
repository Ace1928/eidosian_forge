import errno
import os
from functools import partial
from threading import Thread
from pyudev._os import pipe, poll
from pyudev._util import eintr_retry_call, ensure_byte_string
from pyudev.device import Device
@classmethod
def from_netlink(cls, context, source='udev'):
    """
        Create a monitor by connecting to the kernel daemon through netlink.

        ``context`` is the :class:`Context` to use.  ``source`` is a string,
        describing the event source.  Two sources are available:

        ``'udev'`` (the default)
          Events emitted after udev as registered and configured the device.
          This is the absolutely recommended source for applications.

        ``'kernel'``
          Events emitted directly after the kernel has seen the device.  The
          device has not yet been configured by udev and might not be usable
          at all.  **Never** use this, unless you know what you are doing.

        Return a new :class:`Monitor` object, which is connected to the
        given source.  Raise :exc:`~exceptions.ValueError`, if an invalid
        source has been specified.  Raise
        :exc:`~exceptions.EnvironmentError`, if the creation of the monitor
        failed.
        """
    if source not in ('kernel', 'udev'):
        raise ValueError('Invalid source: {0!r}. Must be one of "udev" or "kernel"'.format(source))
    monitor = context._libudev.udev_monitor_new_from_netlink(context, ensure_byte_string(source))
    if not monitor:
        raise EnvironmentError('Could not create udev monitor')
    return cls(context, monitor)