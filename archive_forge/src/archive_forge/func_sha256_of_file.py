from collections import namedtuple
from hashlib import sha256
import os
import shutil
import sys
import fnmatch
from sympy.testing.pytest import XFAIL
def sha256_of_file(path, nblocks=128):
    """ Computes the SHA256 hash of a file.

    Parameters
    ==========

    path : string
        Path to file to compute hash of.
    nblocks : int
        Number of blocks to read per iteration.

    Returns
    =======

    hashlib sha256 hash object. Use ``.digest()`` or ``.hexdigest()``
    on returned object to get binary or hex encoded string.
    """
    sh = sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(nblocks * sh.block_size), b''):
            sh.update(chunk)
    return sh