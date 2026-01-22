import hashlib
import os
from rsa._compat import range
from rsa import common, transform, core
def yield_fixedblocks(infile, blocksize):
    """Generator, yields each block of ``blocksize`` bytes in the input file.

    :param infile: file to read and separate in blocks.
    :param blocksize: block size in bytes.
    :returns: a generator that yields the contents of each block
    """
    while True:
        block = infile.read(blocksize)
        read_bytes = len(block)
        if read_bytes == 0:
            break
        yield block
        if read_bytes < blocksize:
            break