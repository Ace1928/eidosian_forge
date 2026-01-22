from __future__ import annotations
import io
import os
import re
import subprocess
import sys
import tempfile
from . import Image, ImageFile
from ._binary import i32le as i32
from ._deprecate import deprecate
def has_ghostscript():
    global gs_binary, gs_windows_binary
    if gs_binary is None:
        if sys.platform.startswith('win'):
            if gs_windows_binary is None:
                import shutil
                for binary in ('gswin32c', 'gswin64c', 'gs'):
                    if shutil.which(binary) is not None:
                        gs_windows_binary = binary
                        break
                else:
                    gs_windows_binary = False
            gs_binary = gs_windows_binary
        else:
            try:
                subprocess.check_call(['gs', '--version'], stdout=subprocess.DEVNULL)
                gs_binary = 'gs'
            except OSError:
                gs_binary = False
    return gs_binary is not False