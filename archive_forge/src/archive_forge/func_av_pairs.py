import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
@property
def av_pairs(self) -> 'TargetInfo':
    """The target info AV_PAIR structures."""
    return TargetInfo.unpack(self._data[28:].tobytes())