import contextlib
from datetime import datetime
import sys
import time
def readable_bytes_string(bytes):
    """Get a human-readable string for number of bytes."""
    if bytes >= 2 ** 20:
        return '%.1f MB' % (float(bytes) / 2 ** 20)
    elif bytes >= 2 ** 10:
        return '%.1f kB' % (float(bytes) / 2 ** 10)
    else:
        return '%d B' % bytes