from __future__ import absolute_import
import sys
import time
import serial
from serial.serialutil import  to_bytes
@serial.Serial.rts.setter
def rts(self, level):
    self.formatter.control('RTS', 'active' if level else 'inactive')
    serial.Serial.rts.__set__(self, level)