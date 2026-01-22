from __future__ import annotations
import ctypes
import os
import sys
from functools import lru_cache
from typing import TYPE_CHECKING
from .api import PlatformDirsABC
def get_win_folder_via_ctypes(csidl_name: str) -> str:
    """Get folder with ctypes."""
    csidl_const = {'CSIDL_APPDATA': 26, 'CSIDL_COMMON_APPDATA': 35, 'CSIDL_LOCAL_APPDATA': 28, 'CSIDL_PERSONAL': 5, 'CSIDL_MYPICTURES': 39, 'CSIDL_MYVIDEO': 14, 'CSIDL_MYMUSIC': 13, 'CSIDL_DOWNLOADS': 40}.get(csidl_name)
    if csidl_const is None:
        msg = f'Unknown CSIDL name: {csidl_name}'
        raise ValueError(msg)
    buf = ctypes.create_unicode_buffer(1024)
    windll = getattr(ctypes, 'windll')
    windll.shell32.SHGetFolderPathW(None, csidl_const, None, 0, buf)
    if any((ord(c) > 255 for c in buf)):
        buf2 = ctypes.create_unicode_buffer(1024)
        if windll.kernel32.GetShortPathNameW(buf.value, buf2, 1024):
            buf = buf2
    if csidl_name == 'CSIDL_DOWNLOADS':
        return os.path.join(buf.value, 'Downloads')
    return buf.value