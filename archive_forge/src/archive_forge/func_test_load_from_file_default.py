from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_load_from_file_default(self):
    font_name = pygame_font.get_default_font()
    font_path = os.path.join(os.path.split(pygame.__file__)[0], pygame_font.get_default_font())
    f = pygame_font.Font(font_path)