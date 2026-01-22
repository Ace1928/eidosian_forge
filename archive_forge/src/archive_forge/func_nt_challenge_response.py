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
def nt_challenge_response(self) -> typing.Optional[bytes]:
    """The NtChallengeResponse or None if not set."""
    return _unpack_payload(self._data, 20)