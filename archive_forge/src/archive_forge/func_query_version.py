import _thread
import struct
import sys
import time
from collections import deque
from io import BytesIO
from fastbencode import bdecode_as_tuple, bencode
import breezy
from ... import debug, errors, osutils
from ...trace import log_exception_quietly, mutter
from . import message, request
def query_version(self):
    """Return protocol version number of the server."""
    self.call(b'hello')
    resp = self.read_response_tuple()
    if resp == (b'ok', b'1'):
        return 1
    elif resp == (b'ok', b'2'):
        return 2
    else:
        raise errors.SmartProtocolError('bad response {!r}'.format(resp))