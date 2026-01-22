import array
import os
import struct
import six
from ._exceptions import *
from ._utils import validate_utf8
from threading import Lock
@staticmethod
def create_frame(data, opcode, fin=1):
    """
        create frame to send text, binary and other data.

        data: data to send. This is string value(byte array).
            if opcode is OPCODE_TEXT and this value is unicode,
            data value is converted into unicode string, automatically.

        opcode: operation code. please see OPCODE_XXX.

        fin: fin flag. if set to 0, create continue fragmentation.
        """
    if opcode == ABNF.OPCODE_TEXT and isinstance(data, six.text_type):
        data = data.encode('utf-8')
    return ABNF(fin, 0, 0, 0, opcode, 1, data)