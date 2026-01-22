from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_get_descent(self):
    f = pygame_font.Font(None, 20)
    descent = f.get_descent()
    self.assertTrue(isinstance(descent, int))
    self.assertTrue(descent < 0)