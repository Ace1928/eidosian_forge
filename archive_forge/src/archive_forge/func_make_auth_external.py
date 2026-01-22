from binascii import hexlify
from enum import Enum
import os
from typing import Optional
def make_auth_external() -> bytes:
    """Prepare an AUTH command line with the current effective user ID.

    This is the preferred authentication method for typical D-Bus connections
    over a Unix domain socket.
    """
    hex_uid = hexlify(str(os.geteuid()).encode('ascii'))
    return b'AUTH EXTERNAL %b\r\n' % hex_uid