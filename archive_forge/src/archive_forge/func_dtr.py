from __future__ import absolute_import
import sys
import time
import serial
from serial.serialutil import  to_bytes
@serial.Serial.dtr.setter
def dtr(self, level):
    self.formatter.control('DTR', 'active' if level else 'inactive')
    serial.Serial.dtr.__set__(self, level)