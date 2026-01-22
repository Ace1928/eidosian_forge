from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_set_underline_property(self):
    f = pygame_font.Font(None, 20)
    self.assertFalse(f.underline)
    f.underline = True
    self.assertTrue(f.underline)
    f.underline = False
    self.assertFalse(f.underline)