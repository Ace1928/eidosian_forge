import errno
import os
import sys
from stat import S_IMODE, S_ISDIR, ST_MODE
from .. import osutils, transport, urlutils
def writer(fd):
    if raw_bytes:
        os.write(fd, raw_bytes)