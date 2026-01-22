import time
import os
import libevdev
from ._clib import Libevdev, UinputDevice
from ._clib import READ_FLAG_SYNC, READ_FLAG_NORMAL, READ_FLAG_FORCE_SYNC, READ_FLAG_BLOCKING
from .event import InputEvent
from .const import InputProperty
def set_leds(self, leds):
    """
        Write the LEDs to the device::

             >>> fd = open(path, 'r+b', buffering=0)
             >>> d = libevdev.Device(fd)
             >>> d.set_leds([(libevdev.EV_LED.LED_NUML, 0),
                             (libevdev.EV_LED.LED_SCROLLL, 1)])

        Updating LED states require the fd to be in write-mode.
        """
    for led in leds:
        if led[0].type is not libevdev.EV_LED:
            raise InvalidArgumentException('Event code must be one of EV_LED.*')
    for led in leds:
        self._libevdev.set_led(led[0].value, led[1])