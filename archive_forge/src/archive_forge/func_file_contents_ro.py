import binascii
import os
import mmap
import sys
import time
import errno
from io import BytesIO
from smmap import (
import hashlib
from gitdb.const import (
def file_contents_ro(fd, stream=False, allow_mmap=True):
    """:return: read-only contents of the file represented by the file descriptor fd

    :param fd: file descriptor opened for reading
    :param stream: if False, random access is provided, otherwise the stream interface
        is provided.
    :param allow_mmap: if True, its allowed to map the contents into memory, which
        allows large files to be handled and accessed efficiently. The file-descriptor
        will change its position if this is False"""
    try:
        if allow_mmap:
            try:
                return mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            except OSError:
                return mmap.mmap(fd, os.fstat(fd).st_size, access=mmap.ACCESS_READ)
    except OSError:
        pass
    contents = os.read(fd, os.fstat(fd).st_size)
    if stream:
        return _RandomAccessBytesIO(contents)
    return contents