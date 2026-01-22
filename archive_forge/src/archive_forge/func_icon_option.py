import logging
import os
import sys
from pathlib import Path
from typing import List
from . import MAJOR_VERSION, run_command
@staticmethod
def icon_option():
    if sys.platform == 'linux':
        return '--linux-icon'
    elif sys.platform == 'win32':
        return '--windows-icon-from-ico'
    else:
        return '--macos-app-icon'