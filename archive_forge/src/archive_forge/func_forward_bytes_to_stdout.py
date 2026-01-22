import os
import sys
import subprocess
import locale
import warnings
from numpy.distutils.misc_util import is_sequence, make_temp_file
from numpy.distutils import log
def forward_bytes_to_stdout(val):
    """
    Forward bytes from a subprocess call to the console, without attempting to
    decode them.

    The assumption is that the subprocess call already returned bytes in
    a suitable encoding.
    """
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout.buffer.write(val)
    elif hasattr(sys.stdout, 'encoding'):
        sys.stdout.write(val.decode(sys.stdout.encoding))
    else:
        sys.stdout.write(val.decode('utf8', errors='replace'))