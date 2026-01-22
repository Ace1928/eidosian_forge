import xcffib
import struct
import io
from . import xproto
@classmethod
def synthetic(cls, timestamp, power_level, state):
    self = cls.__new__(cls)
    self.timestamp = timestamp
    self.power_level = power_level
    self.state = state
    return self