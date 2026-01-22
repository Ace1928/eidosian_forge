import os
import pathlib
import platform
import stat
import sys
from logging import getLogger
from typing import Union
def is_windows_unc_path(path) -> bool:
    return sys.platform == 'win32' and path.drive.startswith('\\\\')