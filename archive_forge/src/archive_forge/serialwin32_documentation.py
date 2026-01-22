from __future__ import absolute_import
import ctypes
import time
from serial import win32
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, PortNotOpenError, SerialTimeoutException
Change the exclusive access setting.