import array
import binascii
import io
import os
import tempfile
import unittest
import glob
import pathlib
from pygame.tests.test_utils import example_path, png, tostring
import pygame, pygame.image, pygame.pkgdata
def test_fromstring__and_tostring(self):
    """Ensure methods tostring() and fromstring() are symmetric."""
    import itertools
    fmts = ('RGBA', 'ARGB', 'BGRA')
    fmt_permutations = itertools.permutations(fmts, 2)
    fmt_combinations = itertools.combinations(fmts, 2)

    def convert(fmt1, fmt2, str_buf):
        pos_fmt1 = {k: v for v, k in enumerate(fmt1)}
        pos_fmt2 = {k: v for v, k in enumerate(fmt2)}
        byte_buf = array.array('B', str_buf)
        num_quads = len(byte_buf) // 4
        for i in range(num_quads):
            i4 = i * 4
            R = byte_buf[i4 + pos_fmt1['R']]
            G = byte_buf[i4 + pos_fmt1['G']]
            B = byte_buf[i4 + pos_fmt1['B']]
            A = byte_buf[i4 + pos_fmt1['A']]
            byte_buf[i4 + pos_fmt2['R']] = R
            byte_buf[i4 + pos_fmt2['G']] = G
            byte_buf[i4 + pos_fmt2['B']] = B
            byte_buf[i4 + pos_fmt2['A']] = A
        return tostring(byte_buf)
    test_surface = pygame.Surface((64, 256), flags=pygame.SRCALPHA, depth=32)
    for i in range(256):
        for j in range(16):
            intensity = j * 16 + 15
            test_surface.set_at((j + 0, i), (intensity, i, i, i))
            test_surface.set_at((j + 16, i), (i, intensity, i, i))
            test_surface.set_at((j + 32, i), (i, i, intensity, i))
            test_surface.set_at((j + 32, i), (i, i, i, intensity))
    self._assertSurfaceEqual(test_surface, test_surface, 'failing with identical surfaces')
    for pair in fmt_combinations:
        fmt1_buf = pygame.image.tostring(test_surface, pair[0])
        fmt1_convert_buf = convert(pair[1], pair[0], convert(pair[0], pair[1], fmt1_buf))
        test_convert_two_way = pygame.image.fromstring(fmt1_convert_buf, test_surface.get_size(), pair[0])
        self._assertSurfaceEqual(test_surface, test_convert_two_way, f'converting {pair[0]} to {pair[1]} and back is not symmetric')
    for pair in fmt_permutations:
        fmt1_buf = pygame.image.tostring(test_surface, pair[0])
        fmt2_convert_buf = convert(pair[0], pair[1], fmt1_buf)
        test_convert_one_way = pygame.image.fromstring(fmt2_convert_buf, test_surface.get_size(), pair[1])
        self._assertSurfaceEqual(test_surface, test_convert_one_way, f'converting {pair[0]} to {pair[1]} failed')
    for fmt in fmts:
        test_buf = pygame.image.tostring(test_surface, fmt)
        test_to_from_fmt_string = pygame.image.fromstring(test_buf, test_surface.get_size(), fmt)
        self._assertSurfaceEqual(test_surface, test_to_from_fmt_string, f"tostring/fromstring functions are not symmetric with '{fmt}' format")