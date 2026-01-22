import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_cache(self):
    glyphs = 'abcde'
    glen = len(glyphs)
    other_glyphs = '123'
    oglen = len(other_glyphs)
    uempty = ''
    many_glyphs = uempty.join([chr(i) for i in range(32, 127)])
    mglen = len(many_glyphs)
    count = 0
    access = 0
    hit = 0
    miss = 0
    f = ft.Font(None, size=24, font_index=0, resolution=72, ucs4=False)
    f.style = ft.STYLE_NORMAL
    f.antialiased = True
    self.assertEqual(f._debug_cache_stats, (0, 0, 0, 0, 0))
    count = access = miss = glen
    f.render_raw(glyphs)
    self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
    access += glen
    hit += glen
    f.vertical = True
    f.render_raw(glyphs)
    f.vertical = False
    self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
    count += oglen
    access += oglen
    miss += oglen
    f.render_raw(other_glyphs)
    self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
    count += glen
    access += glen
    miss += glen
    f.render_raw(glyphs, size=12)
    self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
    access += oglen
    hit += oglen
    f.underline = True
    f.render_raw(other_glyphs)
    f.underline = False
    self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
    count += glen
    access += glen
    miss += glen
    f.oblique = True
    f.render_raw(glyphs)
    f.oblique = False
    self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
    count += glen
    access += glen
    miss += glen
    f.strong = True
    f.render_raw(glyphs)
    f.strong = False
    ccount, cdelete_count, caccess, chit, cmiss = f._debug_cache_stats
    self.assertEqual((ccount + cdelete_count, caccess, chit, cmiss), (count, access, hit, miss))
    count += glen
    access += glen
    miss += glen
    f.render_raw(glyphs, rotation=10)
    ccount, cdelete_count, caccess, chit, cmiss = f._debug_cache_stats
    self.assertEqual((ccount + cdelete_count, caccess, chit, cmiss), (count, access, hit, miss))
    count += oglen
    access += oglen
    miss += oglen
    f.antialiased = False
    f.render_raw(other_glyphs)
    f.antialiased = True
    ccount, cdelete_count, caccess, chit, cmiss = f._debug_cache_stats
    self.assertEqual((ccount + cdelete_count, caccess, chit, cmiss), (count, access, hit, miss))
    count += 2 * mglen
    access += 2 * mglen
    miss += 2 * mglen
    f.get_metrics(many_glyphs, size=8)
    f.get_metrics(many_glyphs, size=10)
    ccount, cdelete_count, caccess, chit, cmiss = f._debug_cache_stats
    self.assertTrue(ccount < count)
    self.assertEqual((ccount + cdelete_count, caccess, chit, cmiss), (count, access, hit, miss))