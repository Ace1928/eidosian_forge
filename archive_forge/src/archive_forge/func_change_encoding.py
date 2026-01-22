from __future__ import absolute_import
import codecs
import os
import sys
import threading
import serial
from serial.tools.list_ports import comports
from serial.tools import hexlify_codec
def change_encoding(self):
    """change encoding on the serial port"""
    sys.stderr.write('\n--- Enter new encoding name [{}]: '.format(self.input_encoding))
    with self.console:
        new_encoding = sys.stdin.readline().strip()
    if new_encoding:
        try:
            codecs.lookup(new_encoding)
        except LookupError:
            sys.stderr.write('--- invalid encoding name: {}\n'.format(new_encoding))
        else:
            self.set_rx_encoding(new_encoding)
            self.set_tx_encoding(new_encoding)
    sys.stderr.write('--- serial input encoding: {}\n'.format(self.input_encoding))
    sys.stderr.write('--- serial output encoding: {}\n'.format(self.output_encoding))