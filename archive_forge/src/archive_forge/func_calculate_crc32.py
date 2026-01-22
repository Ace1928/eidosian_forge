import ctypes
import hashlib
import os
import pathlib
import platform
import sys
import time as _time
import zlib
from datetime import datetime, timedelta, timezone, tzinfo
from typing import BinaryIO, List, Optional, Union
import py7zr.win32compat
from py7zr import Bad7zFile
from py7zr.win32compat import is_windows_native_python, is_windows_unc_path
def calculate_crc32(data: bytes, value: int=0, blocksize: int=1024 * 1024) -> int:
    """Calculate CRC32 of strings with arbitrary lengths."""
    if len(data) <= blocksize:
        value = zlib.crc32(data, value)
    else:
        length = len(data)
        pos = blocksize
        value = zlib.crc32(data[:pos], value)
        while pos < length:
            value = zlib.crc32(data[pos:pos + blocksize], value)
            pos += blocksize
    return value & 4294967295