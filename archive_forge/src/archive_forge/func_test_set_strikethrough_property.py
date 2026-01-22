from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_set_strikethrough_property(self):
    if pygame_font.__name__ != 'pygame.ftfont':
        f = pygame_font.Font(None, 20)
        self.assertFalse(f.strikethrough)
        f.strikethrough = True
        self.assertTrue(f.strikethrough)
        f.strikethrough = False
        self.assertFalse(f.strikethrough)