from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_bold_strikethrough(self):
    if pygame_font.__name__ != 'pygame.ftfont':
        self.assertTrue(self.query(bold=True, strikethrough=True))