from __future__ import absolute_import
import codecs
import os
import sys
import threading
import serial
from serial.tools.list_ports import comports
from serial.tools import hexlify_codec
def set_rx_encoding(self, encoding, errors='replace'):
    """set encoding for received data"""
    self.input_encoding = encoding
    self.rx_decoder = codecs.getincrementaldecoder(encoding)(errors)