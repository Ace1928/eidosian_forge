from binascii import hexlify
from enum import Enum
import os
from typing import Optional
def make_auth_anonymous() -> bytes:
    """Format an AUTH command line for the ANONYMOUS mechanism

    Jeepney's higher-level wrappers don't currently use this mechanism,
    but third-party code may choose to.

    See <https://tools.ietf.org/html/rfc4505> for details.
    """
    from . import __version__
    trace = hexlify(('Jeepney %s' % __version__).encode('ascii'))
    return b'AUTH ANONYMOUS %s\r\n' % trace