import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_kerning__enabled(self):
    """Ensures exceptions are not raised when calling freetype methods
        while kerning is enabled.

        Note: This does not test what changes occur to a rendered font by
              having kerning enabled.

        Related to issue #367.
        """
    surface = pygame.Surface((10, 10), 0, 32)
    TEST_TEXT = 'Freetype Font'
    ft_font = self._TEST_FONTS['bmp-8-75dpi']
    ft_font.kerning = True
    metrics = ft_font.get_metrics(TEST_TEXT)
    self.assertIsInstance(metrics, list)
    rect = ft_font.get_rect(TEST_TEXT)
    self.assertIsInstance(rect, pygame.Rect)
    font_surf, rect = ft_font.render(TEST_TEXT)
    self.assertIsInstance(font_surf, pygame.Surface)
    self.assertIsInstance(rect, pygame.Rect)
    rect = ft_font.render_to(surface, (0, 0), TEST_TEXT)
    self.assertIsInstance(rect, pygame.Rect)
    buf, size = ft_font.render_raw(TEST_TEXT)
    self.assertIsInstance(buf, bytes)
    self.assertIsInstance(size, tuple)
    rect = ft_font.render_raw_to(surface.get_view('2'), TEST_TEXT)
    self.assertIsInstance(rect, pygame.Rect)