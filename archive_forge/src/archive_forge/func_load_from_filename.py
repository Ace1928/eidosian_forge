import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def load_from_filename(self, filename=None):
    if filename is None:
        filename = self._filename
    create_new = False
    read_only = True
    keep_cache_in_memory = False
    with self._fi as lib:
        multibitmap = lib.FreeImage_OpenMultiBitmap(self._ftype, efn(filename), create_new, read_only, keep_cache_in_memory, self._flags)
        multibitmap = ctypes.c_void_p(multibitmap)
        if not multibitmap:
            err = self._fi._get_error_message()
            raise ValueError('Could not open file "%s" as multi-image: %s' % (self._filename, err))
        self._set_bitmap(multibitmap, (lib.FreeImage_CloseMultiBitmap, multibitmap))