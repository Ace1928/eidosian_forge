from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_issue_742(self):
    """that the font background does not crash."""
    surf = pygame.Surface((320, 240))
    font = pygame_font.Font(None, 24)
    image = font.render('Test', 0, (255, 255, 255), (0, 0, 0))
    self.assertIsNone(image.get_colorkey())
    image.set_alpha(255)
    surf.blit(image, (0, 0))
    self.assertEqual(surf.get_at((0, 0)), pygame.Color(0, 0, 0))