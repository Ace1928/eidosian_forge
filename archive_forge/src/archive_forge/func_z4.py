import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
@z4.setter
def z4(self, value: int) -> None:
    self._data[4:8] = struct.pack('<I', value)