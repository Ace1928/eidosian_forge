import os
import abc
import sys
from Cryptodome.Util.py3compat import byte_string
from Cryptodome.Util._file_system import pycryptodome_filename
def is_writeable_buffer(x):
    return isinstance(x, bytearray) or (isinstance(x, memoryview) and (not x.readonly))