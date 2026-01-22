import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_render_raw_to(self):
    font = self._TEST_FONTS['sans']
    text = 'abc'
    srect = font.get_rect(text, size=24)
    surf = pygame.Surface(srect.size, 0, 8)
    rrect = font.render_raw_to(surf.get_view('2'), text, size=24)
    self.assertEqual(rrect, srect)
    for bpp in [24, 32]:
        surf = pygame.Surface(srect.size, 0, bpp)
        rrect = font.render_raw_to(surf.get_view('r'), text, size=24)
        self.assertEqual(rrect, srect)
    srect = font.get_rect(text, size=24, style=ft.STYLE_UNDERLINE)
    surf = pygame.Surface(srect.size, 0, 8)
    rrect = font.render_raw_to(surf.get_view('2'), text, size=24, style=ft.STYLE_UNDERLINE)
    self.assertEqual(rrect, srect)
    for bpp in [24, 32]:
        surf = pygame.Surface(srect.size, 0, bpp)
        rrect = font.render_raw_to(surf.get_view('r'), text, size=24, style=ft.STYLE_UNDERLINE)
        self.assertEqual(rrect, srect)
    font.antialiased = False
    try:
        srect = font.get_rect(text, size=24)
        surf = pygame.Surface(srect.size, 0, 8)
        rrect = font.render_raw_to(surf.get_view('2'), text, size=24)
        self.assertEqual(rrect, srect)
        for bpp in [24, 32]:
            surf = pygame.Surface(srect.size, 0, bpp)
            rrect = font.render_raw_to(surf.get_view('r'), text, size=24)
            self.assertEqual(rrect, srect)
    finally:
        font.antialiased = True
    srect = font.get_rect(text, size=24)
    for bpp in [16, 24, 32]:
        surf = pygame.Surface(srect.size, 0, bpp)
        rrect = font.render_raw_to(surf.get_view('2'), text, size=24)
        self.assertEqual(rrect, srect)
    srect = font.get_rect(text, size=24, style=ft.STYLE_UNDERLINE)
    for bpp in [16, 24, 32]:
        surf = pygame.Surface(srect.size, 0, bpp)
        rrect = font.render_raw_to(surf.get_view('2'), text, size=24, style=ft.STYLE_UNDERLINE)
        self.assertEqual(rrect, srect)
    font.antialiased = False
    try:
        srect = font.get_rect(text, size=24)
        for bpp in [16, 24, 32]:
            surf = pygame.Surface(srect.size, 0, bpp)
            rrect = font.render_raw_to(surf.get_view('2'), text, size=24)
            self.assertEqual(rrect, srect)
    finally:
        font.antialiased = True
    srect = font.get_rect(text, size=24)
    surf_buf = pygame.Surface(srect.size, 0, 32).get_view('2')
    for dest in [0, 'a', 'ab', (), (1,), ('a', 2), (1, 'a'), (1 + 2j, 2), (1, 1 + 2j), (1, int), (int, 1)]:
        self.assertRaises(TypeError, font.render_raw_to, surf_buf, text, dest, size=24)