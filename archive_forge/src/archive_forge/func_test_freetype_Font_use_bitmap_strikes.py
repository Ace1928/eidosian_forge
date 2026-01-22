import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_use_bitmap_strikes(self):
    f = self._TEST_FONTS['mono']
    try:
        self.assertTrue(f.use_bitmap_strikes)
        s_strike, sz = f.render_raw('A', size=19)
        try:
            f.vertical = True
            s_strike_vert, sz = f.render_raw('A', size=19)
        finally:
            f.vertical = False
        try:
            f.wide = True
            s_strike_wide, sz = f.render_raw('A', size=19)
        finally:
            f.wide = False
        try:
            f.underline = True
            s_strike_underline, sz = f.render_raw('A', size=19)
        finally:
            f.underline = False
        s_strike_rot45, sz = f.render_raw('A', size=19, rotation=45)
        try:
            f.strong = True
            s_strike_strong, sz = f.render_raw('A', size=19)
        finally:
            f.strong = False
        try:
            f.oblique = True
            s_strike_oblique, sz = f.render_raw('A', size=19)
        finally:
            f.oblique = False
        f.use_bitmap_strikes = False
        self.assertFalse(f.use_bitmap_strikes)
        s_outline, sz = f.render_raw('A', size=19)
        self.assertNotEqual(s_outline, s_strike)
        try:
            f.vertical = True
            s_outline, sz = f.render_raw('A', size=19)
            self.assertNotEqual(s_outline, s_strike_vert)
        finally:
            f.vertical = False
        try:
            f.wide = True
            s_outline, sz = f.render_raw('A', size=19)
            self.assertNotEqual(s_outline, s_strike_wide)
        finally:
            f.wide = False
        try:
            f.underline = True
            s_outline, sz = f.render_raw('A', size=19)
            self.assertNotEqual(s_outline, s_strike_underline)
        finally:
            f.underline = False
        s_outline, sz = f.render_raw('A', size=19, rotation=45)
        self.assertEqual(s_outline, s_strike_rot45)
        try:
            f.strong = True
            s_outline, sz = f.render_raw('A', size=19)
            self.assertEqual(s_outline, s_strike_strong)
        finally:
            f.strong = False
        try:
            f.oblique = True
            s_outline, sz = f.render_raw('A', size=19)
            self.assertEqual(s_outline, s_strike_oblique)
        finally:
            f.oblique = False
    finally:
        f.use_bitmap_strikes = True