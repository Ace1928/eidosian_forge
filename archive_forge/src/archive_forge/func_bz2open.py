from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
@classmethod
def bz2open(cls, name, mode='r', fileobj=None, compresslevel=9, **kwargs):
    """Open bzip2 compressed tar archive name for reading or writing.
           Appending is not allowed.
        """
    if mode not in ('r', 'w', 'x'):
        raise ValueError("mode must be 'r', 'w' or 'x'")
    try:
        from bz2 import BZ2File
    except ImportError:
        raise CompressionError('bz2 module is not available') from None
    fileobj = BZ2File(fileobj or name, mode, compresslevel=compresslevel)
    try:
        t = cls.taropen(name, mode, fileobj, **kwargs)
    except (OSError, EOFError) as e:
        fileobj.close()
        if mode == 'r':
            raise ReadError('not a bzip2 file') from e
        raise
    except:
        fileobj.close()
        raise
    t._extfileobj = False
    return t