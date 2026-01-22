import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_height(self):
    f = self._TEST_FONTS['sans']
    self.assertEqual(f.height, 2355)
    f = self._TEST_FONTS['fixed']
    self.assertEqual(f.height, 1100)
    self.assertRaises(RuntimeError, lambda: nullfont().height)