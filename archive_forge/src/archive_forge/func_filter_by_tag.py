import errno
import os
from functools import partial
from threading import Thread
from pyudev._os import pipe, poll
from pyudev._util import eintr_retry_call, ensure_byte_string
from pyudev.device import Device
def filter_by_tag(self, tag):
    """
        Filter incoming events by the given ``tag``.

        ``tag`` is a byte or unicode string with the name of a tag.  Only
        events for devices which have this tag attached pass the filter and are
        handed to the caller.

        Like with :meth:`filter_by` this filter is also executed inside the
        kernel, so that client processes are usually not woken up for devices
        without the given ``tag``.

        .. udevversion:: 154

        .. versionadded:: 0.9

        .. versionchanged:: 0.15
           This method can also be after :meth:`start()` now.
        """
    self._libudev.udev_monitor_filter_add_match_tag(self, ensure_byte_string(tag))
    self._libudev.udev_monitor_filter_update(self)