import serial
from serial import (
from twisted.python.runtime import platform

    Base class for Windows and POSIX serial ports.

    @ivar _serialFactory: a pyserial C{serial.Serial} factory, used to create
        the instance stored in C{self._serial}. Overrideable to enable easier
        testing.

    @ivar _serial: a pyserial C{serial.Serial} instance used to manage the
        options on the serial port.
    