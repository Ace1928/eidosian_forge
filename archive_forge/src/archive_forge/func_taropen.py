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
def taropen(cls, name, mode='r', fileobj=None, **kwargs):
    """Open uncompressed tar archive name for reading or writing.
        """
    if mode not in ('r', 'a', 'w', 'x'):
        raise ValueError("mode must be 'r', 'a', 'w' or 'x'")
    return cls(name, mode, fileobj, **kwargs)