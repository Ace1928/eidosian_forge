import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
def test_fill_blend_rgba(self):
    destinations = [self._make_surface(8), self._make_surface(16), self._make_surface(16, srcalpha=True), self._make_surface(24), self._make_surface(32), self._make_surface(32, srcalpha=True)]
    blend = [('BLEND_RGBA_ADD', (0, 25, 100, 255), lambda a, b: min(a + b, 255)), ('BLEND_RGBA_SUB', (0, 25, 100, 255), lambda a, b: max(a - b, 0)), ('BLEND_RGBA_MULT', (0, 7, 100, 255), lambda a, b: a * b + 255 >> 8), ('BLEND_RGBA_MIN', (0, 255, 0, 255), min), ('BLEND_RGBA_MAX', (0, 255, 0, 255), max)]
    for dst in destinations:
        dst_palette = [dst.unmap_rgb(dst.map_rgb(c)) for c in self._test_palette]
        for blend_name, fill_color, op in blend:
            fc = dst.unmap_rgb(dst.map_rgb(fill_color))
            self._fill_surface(dst)
            p = []
            for dc in dst_palette:
                c = [op(dc[i], fc[i]) for i in range(4)]
                if not dst.get_masks()[3]:
                    c[3] = 255
                c = dst.unmap_rgb(dst.map_rgb(c))
                p.append(c)
            dst.fill(fill_color, special_flags=getattr(pygame, blend_name))
            self._assert_surface(dst, p, f', {blend_name}')