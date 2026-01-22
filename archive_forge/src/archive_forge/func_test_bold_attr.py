from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_bold_attr(self):
    f = pygame_font.Font(None, 20)
    self.assertFalse(f.bold)
    f.bold = True
    self.assertTrue(f.bold)
    f.bold = False
    self.assertFalse(f.bold)