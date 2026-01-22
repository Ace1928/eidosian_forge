import unittest
import pygame.constants
def test_k__type(self):
    """Ensures K constants are the correct type."""
    for name in self.K_NAMES:
        value = getattr(pygame.constants, name)
        self.assertIs(type(value), int)