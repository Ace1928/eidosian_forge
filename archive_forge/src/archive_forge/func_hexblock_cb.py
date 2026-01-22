import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@classmethod
def hexblock_cb(cls, callback, data, address=None, bits=None, width=16, cb_args=(), cb_kwargs={}):
    """
        Dump a block of binary data using a callback function to convert each
        line of text.

        @type  callback: function
        @param callback: Callback function to convert each line of data.

        @type  data: str
        @param data: Binary data.

        @type  address: str
        @param address:
            (Optional) Memory address where the data was read from.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @type  cb_args: str
        @param cb_args:
            (Optional) Arguments to pass to the callback function.

        @type  cb_kwargs: str
        @param cb_kwargs:
            (Optional) Keyword arguments to pass to the callback function.

        @type  width: int
        @param width:
            (Optional) Maximum number of bytes to convert per text line.

        @rtype:  str
        @return: Multiline output text.
        """
    result = ''
    if address is None:
        for i in compat.xrange(0, len(data), width):
            result = '%s%s\n' % (result, callback(data[i:i + width], *cb_args, **cb_kwargs))
    else:
        for i in compat.xrange(0, len(data), width):
            result = '%s%s: %s\n' % (result, cls.address(address, bits), callback(data[i:i + width], *cb_args, **cb_kwargs))
            address += width
    return result