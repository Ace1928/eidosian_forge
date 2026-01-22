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
def workstation(self) -> typing.Optional[str]:
    """The name of the computer to which the user is logged on."""
    return to_text(_unpack_payload(self._data, 44), encoding=self._encoding, nonstring='passthru')