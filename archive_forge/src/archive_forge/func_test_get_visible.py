import unittest
import os
import platform
import warnings
import pygame
def test_get_visible(self):
    """Ensures get_visible works correctly."""
    for expected_value in (False, True):
        pygame.mouse.set_visible(expected_value)
        visible = pygame.mouse.get_visible()
        self.assertEqual(visible, expected_value)