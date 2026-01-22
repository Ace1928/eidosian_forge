import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
@unittest.skipIf(IS_PYPY, "PyPy doesn't use refcounting")
def test_freetype_Font_dealloc(self):
    import sys
    handle = open(self._sans_path, 'rb')

    def load_font():
        tempFont = ft.Font(handle)
    try:
        load_font()
        self.assertEqual(sys.getrefcount(handle), 2)
    finally:
        handle.close()