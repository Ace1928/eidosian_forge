from __future__ import unicode_literals
import struct
from six import int2byte, binary_type, iterbytes
from .log import logger
def received_data(self, data):
    self.data_received_callback(data)