import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_get_rect(self):
    font = self._TEST_FONTS['sans']

    def test_rect(r):
        self.assertIsInstance(r, pygame.Rect)
    rect_default = font.get_rect('ABCDabcd', size=24)
    test_rect(rect_default)
    self.assertTrue(rect_default.size > (0, 0))
    self.assertTrue(rect_default.width > rect_default.height)
    rect_bigger = font.get_rect('ABCDabcd', size=32)
    test_rect(rect_bigger)
    self.assertTrue(rect_bigger.size > rect_default.size)
    rect_strong = font.get_rect('ABCDabcd', size=24, style=ft.STYLE_STRONG)
    test_rect(rect_strong)
    self.assertTrue(rect_strong.size > rect_default.size)
    font.vertical = True
    rect_vert = font.get_rect('ABCDabcd', size=24)
    test_rect(rect_vert)
    self.assertTrue(rect_vert.width < rect_vert.height)
    font.vertical = False
    rect_oblique = font.get_rect('ABCDabcd', size=24, style=ft.STYLE_OBLIQUE)
    test_rect(rect_oblique)
    self.assertTrue(rect_oblique.width > rect_default.width)
    self.assertTrue(rect_oblique.height == rect_default.height)
    rect_under = font.get_rect('ABCDabcd', size=24, style=ft.STYLE_UNDERLINE)
    test_rect(rect_under)
    self.assertTrue(rect_under.width == rect_default.width)
    self.assertTrue(rect_under.height > rect_default.height)
    ufont = self._TEST_FONTS['mono']
    rect_utf32 = ufont.get_rect('𓁹', size=24)
    rect_utf16 = ufont.get_rect('\ud80c\udc79', size=24)
    self.assertEqual(rect_utf16, rect_utf32)
    ufont.ucs4 = True
    try:
        rect_utf16 = ufont.get_rect('\ud80c\udc79', size=24)
    finally:
        ufont.ucs4 = False
    self.assertNotEqual(rect_utf16, rect_utf32)
    self.assertRaises(RuntimeError, nullfont().get_rect, 'a', size=24)
    rect12 = font.get_rect('A', size=12.0)
    rect24 = font.get_rect('A', size=24.0)
    rect_x = font.get_rect('A', size=(24.0, 12.0))
    self.assertEqual(rect_x.width, rect24.width)
    self.assertEqual(rect_x.height, rect12.height)
    rect_y = font.get_rect('A', size=(12.0, 24.0))
    self.assertEqual(rect_y.width, rect12.width)
    self.assertEqual(rect_y.height, rect24.height)