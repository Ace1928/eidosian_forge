from __future__ import absolute_import
import codecs
import os
import sys
import threading
import serial
from serial.tools.list_ports import comports
from serial.tools import hexlify_codec
def set_tx_encoding(self, encoding, errors='replace'):
    """set encoding for transmitted data"""
    self.output_encoding = encoding
    self.tx_encoder = codecs.getincrementalencoder(encoding)(errors)