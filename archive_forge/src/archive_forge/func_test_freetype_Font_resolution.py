import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_resolution(self):
    text = '|'
    resolution = ft.get_default_resolution()
    new_font = ft.Font(self._sans_path, resolution=2 * resolution)
    self.assertEqual(new_font.resolution, 2 * resolution)
    size_normal = self._TEST_FONTS['sans'].get_rect(text, size=24).size
    size_scaled = new_font.get_rect(text, size=24).size
    size_by_2 = size_normal[0] * 2
    self.assertTrue(size_by_2 + 2 >= size_scaled[0] >= size_by_2 - 2, '%i not equal %i' % (size_scaled[1], size_by_2))
    size_by_2 = size_normal[1] * 2
    self.assertTrue(size_by_2 + 2 >= size_scaled[1] >= size_by_2 - 2, '%i not equal %i' % (size_scaled[1], size_by_2))
    new_resolution = resolution + 10
    ft.set_default_resolution(new_resolution)
    try:
        new_font = ft.Font(self._sans_path, resolution=0)
        self.assertEqual(new_font.resolution, new_resolution)
    finally:
        ft.set_default_resolution()