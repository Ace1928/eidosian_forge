import binascii
import os
import posixpath
import stat
import warnings
import zlib
from collections import namedtuple
from hashlib import sha1
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
def valid_hexsha(hex):
    if len(hex) != 40:
        return False
    try:
        binascii.unhexlify(hex)
    except (TypeError, binascii.Error):
        return False
    else:
        return True