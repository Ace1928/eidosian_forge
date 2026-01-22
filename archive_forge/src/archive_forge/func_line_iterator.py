from __future__ import print_function, unicode_literals
import typing
import array
import io
from io import SEEK_CUR, SEEK_SET
from .mode import Mode
def line_iterator(readable_file, size=None):
    """Iterate over the lines of a file.

    Implementation reads each char individually, which is not very
    efficient.

    Yields:
        str: a single line in the file.

    """
    read = readable_file.read
    line = []
    byte = b'1'
    if size is None or size < 0:
        while byte:
            byte = read(1)
            line.append(byte)
            if byte in b'\n':
                yield b''.join(line)
                del line[:]
    else:
        while byte and size:
            byte = read(1)
            size -= len(byte)
            line.append(byte)
            if byte in b'\n' or not size:
                yield b''.join(line)
                del line[:]