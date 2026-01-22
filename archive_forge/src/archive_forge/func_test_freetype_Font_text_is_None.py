import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_text_is_None(self):
    f = ft.Font(self._sans_path, 36)
    f.style = ft.STYLE_NORMAL
    f.rotation = 0
    text = 'ABCD'
    get_rect = f.get_rect(text)
    f.vertical = True
    get_rect_vert = f.get_rect(text)
    f.vertical = True
    r = f.get_rect(None)
    self.assertEqual(r, get_rect_vert)
    f.vertical = False
    r = f.get_rect(None, style=ft.STYLE_WIDE)
    self.assertEqual(r.height, get_rect.height)
    self.assertTrue(r.width > get_rect.width)
    r = f.get_rect(None)
    self.assertEqual(r, get_rect)
    r = f.get_rect(None, rotation=90)
    self.assertEqual(r.width, get_rect.height)
    self.assertEqual(r.height, get_rect.width)
    self.assertRaises(TypeError, f.get_metrics, None)