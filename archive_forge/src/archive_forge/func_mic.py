import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
@mic.setter
def mic(self, value: bytes) -> None:
    if len(value) != 16:
        raise ValueError('NTLM Authenticate MIC must be 16 bytes long')
    mic_offset = self._get_mic_offset()
    if mic_offset:
        self._data[mic_offset:mic_offset + 16] = value
    else:
        raise ValueError('Cannot set MIC on an Authenticate message with no MIC present')