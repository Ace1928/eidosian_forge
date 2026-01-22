from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_set_strikethrough(self):
    if pygame_font.__name__ != 'pygame.ftfont':
        f = pygame_font.Font(None, 20)
        self.assertFalse(f.get_strikethrough())
        f.set_strikethrough(True)
        self.assertTrue(f.get_strikethrough())
        f.set_strikethrough(False)
        self.assertFalse(f.get_strikethrough())