import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_fixed_sizes(self):
    f = self._TEST_FONTS['sans']
    self.assertEqual(f.fixed_sizes, 0)
    f = self._TEST_FONTS['bmp-8-75dpi']
    self.assertEqual(f.fixed_sizes, 1)
    f = self._TEST_FONTS['mono']
    self.assertEqual(f.fixed_sizes, 2)