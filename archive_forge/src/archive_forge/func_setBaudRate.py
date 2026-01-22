import serial
from serial import (
from twisted.python.runtime import platform
def setBaudRate(self, baudrate):
    if hasattr(self._serial, 'setBaudrate'):
        self._serial.setBaudrate(baudrate)
    else:
        self._serial.setBaudRate(baudrate)