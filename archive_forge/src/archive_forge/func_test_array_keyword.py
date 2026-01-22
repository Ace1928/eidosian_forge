import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
@unittest.skipIf(os.environ.get('SDL_AUDIODRIVER') == 'disk', 'this test fails without real sound card')
def test_array_keyword(self):
    try:
        from numpy import array, arange, zeros, int8, uint8, int16, uint16, int32, uint32
    except ImportError:
        self.skipTest('requires numpy')
    freq = 22050
    format_list = [-8, 8, -16, 16]
    channels_list = [1, 2]
    a_lists = {f: [] for f in format_list}
    a32u_mono = arange(0, 256, 1, uint32)
    a16u_mono = a32u_mono.astype(uint16)
    a8u_mono = a32u_mono.astype(uint8)
    au_list_mono = [(1, a) for a in [a8u_mono, a16u_mono, a32u_mono]]
    for format in format_list:
        if format > 0:
            a_lists[format].extend(au_list_mono)
    a32s_mono = arange(-128, 128, 1, int32)
    a16s_mono = a32s_mono.astype(int16)
    a8s_mono = a32s_mono.astype(int8)
    as_list_mono = [(1, a) for a in [a8s_mono, a16s_mono, a32s_mono]]
    for format in format_list:
        if format < 0:
            a_lists[format].extend(as_list_mono)
    a32u_stereo = zeros([a32u_mono.shape[0], 2], uint32)
    a32u_stereo[:, 0] = a32u_mono
    a32u_stereo[:, 1] = 255 - a32u_mono
    a16u_stereo = a32u_stereo.astype(uint16)
    a8u_stereo = a32u_stereo.astype(uint8)
    au_list_stereo = [(2, a) for a in [a8u_stereo, a16u_stereo, a32u_stereo]]
    for format in format_list:
        if format > 0:
            a_lists[format].extend(au_list_stereo)
    a32s_stereo = zeros([a32s_mono.shape[0], 2], int32)
    a32s_stereo[:, 0] = a32s_mono
    a32s_stereo[:, 1] = -1 - a32s_mono
    a16s_stereo = a32s_stereo.astype(int16)
    a8s_stereo = a32s_stereo.astype(int8)
    as_list_stereo = [(2, a) for a in [a8s_stereo, a16s_stereo, a32s_stereo]]
    for format in format_list:
        if format < 0:
            a_lists[format].extend(as_list_stereo)
    for format in format_list:
        for channels in channels_list:
            try:
                mixer.init(freq, format, channels)
            except pygame.error:
                continue
            try:
                __, f, c = mixer.get_init()
                if f != format or c != channels:
                    continue
                for c, a in a_lists[format]:
                    self._test_array_argument(format, a, c == channels)
            finally:
                mixer.quit()