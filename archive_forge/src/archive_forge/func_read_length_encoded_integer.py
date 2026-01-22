from .charset import MBLENGTH
from .constants import FIELD_TYPE, SERVER_STATUS
from . import err
import struct
import sys
def read_length_encoded_integer(self):
    """Read a 'Length Coded Binary' number from the data buffer.

        Length coded numbers can be anywhere from 1 to 9 bytes depending
        on the value of the first byte.
        """
    c = self.read_uint8()
    if c == NULL_COLUMN:
        return None
    if c < UNSIGNED_CHAR_COLUMN:
        return c
    elif c == UNSIGNED_SHORT_COLUMN:
        return self.read_uint16()
    elif c == UNSIGNED_INT24_COLUMN:
        return self.read_uint24()
    elif c == UNSIGNED_INT64_COLUMN:
        return self.read_uint64()