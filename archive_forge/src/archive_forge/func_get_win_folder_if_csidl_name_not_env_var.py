from __future__ import annotations
import ctypes
import os
import sys
from functools import lru_cache
from typing import TYPE_CHECKING
from .api import PlatformDirsABC
def get_win_folder_if_csidl_name_not_env_var(csidl_name: str) -> str | None:
    """Get folder for a CSIDL name that does not exist as an environment variable."""
    if csidl_name == 'CSIDL_PERSONAL':
        return os.path.join(os.path.normpath(os.environ['USERPROFILE']), 'Documents')
    if csidl_name == 'CSIDL_DOWNLOADS':
        return os.path.join(os.path.normpath(os.environ['USERPROFILE']), 'Downloads')
    if csidl_name == 'CSIDL_MYPICTURES':
        return os.path.join(os.path.normpath(os.environ['USERPROFILE']), 'Pictures')
    if csidl_name == 'CSIDL_MYVIDEO':
        return os.path.join(os.path.normpath(os.environ['USERPROFILE']), 'Videos')
    if csidl_name == 'CSIDL_MYMUSIC':
        return os.path.join(os.path.normpath(os.environ['USERPROFILE']), 'Music')
    return None