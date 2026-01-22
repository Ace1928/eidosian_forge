from __future__ import annotations
import ctypes
import os
import sys
from functools import lru_cache
from typing import TYPE_CHECKING
from .api import PlatformDirsABC
def get_win_folder_from_env_vars(csidl_name: str) -> str:
    """Get folder from environment variables."""
    result = get_win_folder_if_csidl_name_not_env_var(csidl_name)
    if result is not None:
        return result
    env_var_name = {'CSIDL_APPDATA': 'APPDATA', 'CSIDL_COMMON_APPDATA': 'ALLUSERSPROFILE', 'CSIDL_LOCAL_APPDATA': 'LOCALAPPDATA'}.get(csidl_name)
    if env_var_name is None:
        msg = f'Unknown CSIDL name: {csidl_name}'
        raise ValueError(msg)
    result = os.environ.get(env_var_name)
    if result is None:
        msg = f'Unset environment variable: {env_var_name}'
        raise ValueError(msg)
    return result