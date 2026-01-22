from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def create_gnu_header(self, info, encoding, errors):
    """Return the object as a GNU header block sequence.
        """
    info['magic'] = GNU_MAGIC
    buf = b''
    if len(info['linkname'].encode(encoding, errors)) > LENGTH_LINK:
        buf += self._create_gnu_long_header(info['linkname'], GNUTYPE_LONGLINK, encoding, errors)
    if len(info['name'].encode(encoding, errors)) > LENGTH_NAME:
        buf += self._create_gnu_long_header(info['name'], GNUTYPE_LONGNAME, encoding, errors)
    return buf + self._create_header(info, GNU_FORMAT, encoding, errors)