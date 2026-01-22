import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_render_raw(self):
    font = self._TEST_FONTS['sans']
    text = 'abc'
    size = font.get_rect(text, size=24).size
    rend = font.render_raw(text, size=24)
    self.assertIsInstance(rend, tuple)
    self.assertEqual(len(rend), 2)
    r, s = rend
    self.assertIsInstance(r, bytes)
    self.assertIsInstance(s, tuple)
    self.assertTrue(len(s), 2)
    w, h = s
    self.assertIsInstance(w, int)
    self.assertIsInstance(h, int)
    self.assertEqual(s, size)
    self.assertEqual(len(r), w * h)
    r, (w, h) = font.render_raw('', size=24)
    self.assertEqual(w, 0)
    self.assertEqual(h, font.height)
    self.assertEqual(len(r), 0)
    rend = font.render_raw('render_raw', size=24)
    text = ''.join([chr(i) for i in range(31, 64)])
    rend = font.render_raw(text, size=10)