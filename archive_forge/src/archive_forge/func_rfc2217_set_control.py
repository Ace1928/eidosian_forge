from __future__ import absolute_import
import logging
import socket
import struct
import threading
import time
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def rfc2217_set_control(self, value):
    """transmit change of control line to remote"""
    item = self._rfc2217_options['control']
    item.set(value)
    if self._ignore_set_control_answer:
        time.sleep(0.1)
    else:
        item.wait(self._network_timeout)