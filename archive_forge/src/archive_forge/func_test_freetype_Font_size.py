import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_size(self):
    f = ft.Font(None, size=12)
    self.assertEqual(f.size, 12)
    f.size = 22
    self.assertEqual(f.size, 22)
    f.size = 0
    self.assertEqual(f.size, 0)
    f.size = max_point_size
    self.assertEqual(f.size, max_point_size)
    f.size = 6.5
    self.assertEqual(f.size, 6.5)
    f.size = max_point_size_f
    self.assertEqual(f.size, max_point_size_f)
    self.assertRaises(OverflowError, setattr, f, 'size', -1)
    self.assertRaises(OverflowError, setattr, f, 'size', max_point_size + 1)
    f.size = (24.0, 0)
    size = f.size
    self.assertIsInstance(size, float)
    self.assertEqual(size, 24.0)
    f.size = (16, 16)
    size = f.size
    self.assertIsInstance(size, tuple)
    self.assertEqual(len(size), 2)
    x, y = size
    self.assertIsInstance(x, float)
    self.assertEqual(x, 16.0)
    self.assertIsInstance(y, float)
    self.assertEqual(y, 16.0)
    f.size = (20.5, 22.25)
    x, y = f.size
    self.assertEqual(x, 20.5)
    self.assertEqual(y, 22.25)
    f.size = (0, 0)
    size = f.size
    self.assertIsInstance(size, float)
    self.assertEqual(size, 0.0)
    self.assertRaises(ValueError, setattr, f, 'size', (0, 24.0))
    self.assertRaises(TypeError, setattr, f, 'size', (24.0,))
    self.assertRaises(TypeError, setattr, f, 'size', (24.0, 0, 0))
    self.assertRaises(TypeError, setattr, f, 'size', (24j, 24.0))
    self.assertRaises(TypeError, setattr, f, 'size', (24.0, 24j))
    self.assertRaises(OverflowError, setattr, f, 'size', (-1, 16))
    self.assertRaises(OverflowError, setattr, f, 'size', (max_point_size + 1, 16))
    self.assertRaises(OverflowError, setattr, f, 'size', (16, -1))
    self.assertRaises(OverflowError, setattr, f, 'size', (16, max_point_size + 1))
    f75 = self._TEST_FONTS['bmp-18-75dpi']
    sizes = f75.get_sizes()
    self.assertEqual(len(sizes), 1)
    size_pt, width_px, height_px, x_ppem, y_ppem = sizes[0]
    self.assertEqual(size_pt, 18)
    self.assertEqual(x_ppem, 19.0)
    self.assertEqual(y_ppem, 19.0)
    rect = f75.get_rect('A', size=18)
    rect = f75.get_rect('A', size=19)
    rect = f75.get_rect('A', size=(19.0, 19.0))
    self.assertRaises(pygame.error, f75.get_rect, 'A', size=17)
    f100 = self._TEST_FONTS['bmp-18-100dpi']
    sizes = f100.get_sizes()
    self.assertEqual(len(sizes), 1)
    size_pt, width_px, height_px, x_ppem, y_ppem = sizes[0]
    self.assertEqual(size_pt, 18)
    self.assertEqual(x_ppem, 25.0)
    self.assertEqual(y_ppem, 25.0)
    rect = f100.get_rect('A', size=18)
    rect = f100.get_rect('A', size=25)
    rect = f100.get_rect('A', size=(25.0, 25.0))
    self.assertRaises(pygame.error, f100.get_rect, 'A', size=17)