from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_render_ucs2_ucs4(self):
    """that it renders without raising if there is a new enough SDL_ttf."""
    f = pygame_font.Font(None, 20)
    if hasattr(pygame_font, 'UCS4'):
        ucs_2 = '￮'
        s = f.render(ucs_2, False, [0, 0, 0], [255, 255, 255])
        ucs_4 = '𐀀'
        s = f.render(ucs_4, False, [0, 0, 0], [255, 255, 255])