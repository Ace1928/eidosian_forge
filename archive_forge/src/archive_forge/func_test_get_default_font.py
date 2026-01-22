from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_get_default_font(self):
    self.assertEqual(pygame_font.get_default_font(), 'freesansbold.ttf')