from __future__ import absolute_import
import logging
import socket
import struct
import threading
import time
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def rfc2217_flow_server_ready(self):
    """        check if server is ready to receive data. block for some time when
        not.
        """