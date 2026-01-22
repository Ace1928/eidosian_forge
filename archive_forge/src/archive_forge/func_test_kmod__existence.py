import unittest
import pygame.constants
def test_kmod__existence(self):
    """Ensures KMOD constants exist."""
    for name in self.KMOD_CONSTANTS:
        self.assertTrue(hasattr(pygame.constants, name), f'missing constant {name}')