from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_get_sdl_ttf_version(self):

    def test_ver_tuple(ver):
        self.assertIsInstance(ver, tuple)
        self.assertEqual(len(ver), 3)
        for i in ver:
            self.assertIsInstance(i, int)
    if pygame_font.__name__ != 'pygame.ftfont':
        compiled = pygame_font.get_sdl_ttf_version()
        linked = pygame_font.get_sdl_ttf_version(linked=True)
        test_ver_tuple(compiled)
        test_ver_tuple(linked)
        self.assertTrue(linked >= compiled)