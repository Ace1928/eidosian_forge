from __future__ import absolute_import
import sys
import time
import serial
from serial.serialutil import  to_bytes
@serial.Serial.break_condition.setter
def break_condition(self, level):
    self.formatter.control('BRK', 'active' if level else 'inactive')
    serial.Serial.break_condition.__set__(self, level)