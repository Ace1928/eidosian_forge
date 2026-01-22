import contextlib
import errno
import hashlib
import json
import os
import stat
import tempfile
import time
import yaml
from oslo_utils import excutils
def last_bytes(path, num):
    """Return num bytes from the end of the file and unread byte count.

    Returns a tuple containing some content from the file and the
    number of bytes that appear in the file before the point at which
    reading started. The content will be at most ``num`` bytes, taken
    from the end of the file. If the file is smaller than ``num``
    bytes the entire content of the file is returned.

    :param path: The file path to read
    :param num: The number of bytes to return

    :returns: (data, unread_bytes)

    """
    with open(path, 'rb') as fp:
        try:
            fp.seek(-num, os.SEEK_END)
        except IOError as e:
            if e.errno == errno.EINVAL:
                fp.seek(0, os.SEEK_SET)
            else:
                raise
        unread_bytes = fp.tell()
        return (fp.read(), unread_bytes)