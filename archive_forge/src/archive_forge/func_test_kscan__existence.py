import unittest
import pygame.constants
def test_kscan__existence(self):
    """Ensures KSCAN constants exist."""
    for name in self.KSCAN_NAMES:
        self.assertTrue(hasattr(pygame.constants, name), f'missing constant {name}')