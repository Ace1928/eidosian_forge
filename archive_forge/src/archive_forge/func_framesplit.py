from contextlib import contextmanager
import os
import shutil
import tempfile
import struct
def framesplit(bytes):
    """ Split buffer into frames of concatenated chunks

    >>> data = frame(b'Hello') + frame(b'World')
    >>> list(framesplit(data))  # doctest: +SKIP
    [b'Hello', b'World']
    """
    i = 0
    n = len(bytes)
    chunks = list()
    while i < n:
        nbytes = struct.unpack('Q', bytes[i:i + 8])[0]
        i += 8
        yield bytes[i:i + nbytes]
        i += nbytes