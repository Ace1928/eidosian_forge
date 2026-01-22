import sys
import os
import errno
import socket
from botocore.vendored import six
def rename_file(current_filename, new_filename):
    try:
        os.remove(new_filename)
    except OSError as e:
        if not e.errno == errno.ENOENT:
            raise
    os.rename(current_filename, new_filename)