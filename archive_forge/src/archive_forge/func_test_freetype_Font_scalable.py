import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_scalable(self):
    f = self._TEST_FONTS['sans']
    self.assertTrue(f.scalable)
    self.assertRaises(RuntimeError, lambda: nullfont().scalable)