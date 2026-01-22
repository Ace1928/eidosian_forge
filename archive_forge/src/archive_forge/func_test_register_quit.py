import sys
import unittest
import platform
import pygame
def test_register_quit(self):
    """Ensure that a registered function is called on quit()"""
    self.assertEqual(quit_count, 0)
    pygame.init()
    pygame.register_quit(quit_hook)
    pygame.quit()
    self.assertEqual(quit_count, 1)