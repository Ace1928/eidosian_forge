from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_load_from_file_bytes(self):
    font_path = os.path.join(os.path.split(pygame.__file__)[0], pygame_font.get_default_font())
    filesystem_encoding = sys.getfilesystemencoding()
    filesystem_errors = 'replace' if sys.platform == 'win32' else 'surrogateescape'
    try:
        font_path = font_path.decode(filesystem_encoding, filesystem_errors)
    except AttributeError:
        pass
    bfont_path = font_path.encode(filesystem_encoding, filesystem_errors)
    f = pygame_font.Font(bfont_path, 20)