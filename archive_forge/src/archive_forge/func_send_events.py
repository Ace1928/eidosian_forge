import time
import os
import libevdev
from ._clib import Libevdev, UinputDevice
from ._clib import READ_FLAG_SYNC, READ_FLAG_NORMAL, READ_FLAG_FORCE_SYNC, READ_FLAG_BLOCKING
from .event import InputEvent
from .const import InputProperty
def send_events(self, events):
    """
        Send the list of :class:`InputEvent` events through this device. All
        events must have a valid :class:`EventCode` and value, the timestamp
        in the event is ignored and the kernel fills in its own timestamp.

        This function may only be called on a uinput device, not on a normal
        device.

        .. warning::

            an event list must always be terminated with a
            ``libevdev.EV_SYN.SYN_REPORT`` event or the kernel may delay
            processing.

        :param events: a list of :class:`InputEvent` events
        """
    if not self._uinput:
        raise InvalidFileError()
    if None in [e.code for e in events]:
        raise InvalidArgumentException('All events must have an event code')
    if None in [e.value for e in events]:
        raise InvalidArgumentException('All events must have a value')
    for e in events:
        self._uinput.write_event(e.type.value, e.code.value, e.value)