import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_get_sizes(self):
    f = self._TEST_FONTS['sans']
    szlist = f.get_sizes()
    self.assertIsInstance(szlist, list)
    self.assertEqual(len(szlist), 0)
    f = self._TEST_FONTS['bmp-8-75dpi']
    szlist = f.get_sizes()
    self.assertIsInstance(szlist, list)
    self.assertEqual(len(szlist), 1)
    size8 = szlist[0]
    self.assertIsInstance(size8[0], int)
    self.assertEqual(size8[0], 8)
    self.assertIsInstance(size8[1], int)
    self.assertIsInstance(size8[2], int)
    self.assertIsInstance(size8[3], float)
    self.assertEqual(int(size8[3] * 64.0 + 0.5), 8 * 64)
    self.assertIsInstance(size8[4], float)
    self.assertEqual(int(size8[4] * 64.0 + 0.5), 8 * 64)
    f = self._TEST_FONTS['mono']
    szlist = f.get_sizes()
    self.assertIsInstance(szlist, list)
    self.assertEqual(len(szlist), 2)
    size8 = szlist[0]
    self.assertEqual(size8[3], 8)
    self.assertEqual(int(size8[3] * 64.0 + 0.5), 8 * 64)
    self.assertEqual(int(size8[4] * 64.0 + 0.5), 8 * 64)
    size19 = szlist[1]
    self.assertEqual(size19[3], 19)
    self.assertEqual(int(size19[3] * 64.0 + 0.5), 19 * 64)
    self.assertEqual(int(size19[4] * 64.0 + 0.5), 19 * 64)