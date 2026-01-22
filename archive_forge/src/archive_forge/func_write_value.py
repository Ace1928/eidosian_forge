import binascii
import codecs
import datetime
import enum
from io import BytesIO
import itertools
import os
import re
import struct
from xml.parsers.expat import ParserCreate
def write_value(self, value):
    if isinstance(value, str):
        self.simple_element('string', value)
    elif value is True:
        self.simple_element('true')
    elif value is False:
        self.simple_element('false')
    elif isinstance(value, int):
        if -1 << 63 <= value < 1 << 64:
            self.simple_element('integer', '%d' % value)
        else:
            raise OverflowError(value)
    elif isinstance(value, float):
        self.simple_element('real', repr(value))
    elif isinstance(value, dict):
        self.write_dict(value)
    elif isinstance(value, (bytes, bytearray)):
        self.write_bytes(value)
    elif isinstance(value, datetime.datetime):
        self.simple_element('date', _date_to_string(value))
    elif isinstance(value, (tuple, list)):
        self.write_array(value)
    else:
        raise TypeError('unsupported type: %s' % type(value))