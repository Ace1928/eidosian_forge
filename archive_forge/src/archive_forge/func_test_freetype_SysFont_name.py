import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_SysFont_name(self):
    """that SysFont accepts names of various types"""
    fonts = pygame.font.get_fonts()
    size = 12
    font_name = ft.SysFont(fonts[0], size).name
    self.assertFalse(font_name is None)
    names = ','.join(fonts)
    font_name_2 = ft.SysFont(names, size).name
    self.assertEqual(font_name_2, font_name)
    font_name_2 = ft.SysFont(fonts, size).name
    self.assertEqual(font_name_2, font_name)
    names = (name for name in fonts)
    font_name_2 = ft.SysFont(names, size).name
    self.assertEqual(font_name_2, font_name)
    fonts_b = [f.encode() for f in fonts]
    font_name_2 = ft.SysFont(fonts_b[0], size).name
    self.assertEqual(font_name_2, font_name)
    names = b','.join(fonts_b)
    font_name_2 = ft.SysFont(names, size).name
    self.assertEqual(font_name_2, font_name)
    font_name_2 = ft.SysFont(fonts_b, size).name
    self.assertEqual(font_name_2, font_name)
    names = [fonts[0], fonts_b[1], fonts[2], fonts_b[3]]
    font_name_2 = ft.SysFont(names, size).name
    self.assertEqual(font_name_2, font_name)