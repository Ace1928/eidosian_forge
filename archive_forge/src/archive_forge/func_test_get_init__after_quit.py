import sys
import unittest
import platform
import pygame
def test_get_init__after_quit(self):
    pygame.init()
    pygame.quit()
    self.assertFalse(pygame.get_init())