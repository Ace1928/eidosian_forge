import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def load_freeimage(self):
    """Try to load the freeimage lib from the system. If not successful,
        try to download the imageio version and try again.
        """
    success = False
    try:
        self._load_freeimage()
        self._register_api()
        if self.lib.FreeImage_GetVersion().decode('utf-8') >= '3.15':
            success = True
    except OSError:
        pass
    if not success:
        get_freeimage_lib()
        self._load_freeimage()
        self._register_api()
    self.lib.FreeImage_SetOutputMessage(self._error_handler)
    self.lib_version = self.lib.FreeImage_GetVersion().decode('utf-8')