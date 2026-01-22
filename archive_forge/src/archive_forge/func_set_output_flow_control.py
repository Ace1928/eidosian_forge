from __future__ import absolute_import
import errno
import fcntl
import os
import select
import struct
import sys
import termios
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def set_output_flow_control(self, enable=True):
    """        Manually control flow of outgoing data - when hardware or software flow
        control is enabled.
        WARNING: this function is not portable to different platforms!
        """
    if not self.is_open:
        raise PortNotOpenError()
    if enable:
        termios.tcflow(self.fd, termios.TCOON)
    else:
        termios.tcflow(self.fd, termios.TCOOFF)