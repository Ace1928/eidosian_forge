import unittest
import platform
def test_sysfont(self):
    import pygame.font
    pygame.font.init()
    arial = pygame.font.SysFont('Arial', 40)
    self.assertTrue(isinstance(arial, pygame.font.Font))