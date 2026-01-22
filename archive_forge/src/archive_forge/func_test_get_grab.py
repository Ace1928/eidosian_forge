import collections
import time
import unittest
import os
import pygame
@unittest.skip('flaky test, and broken on 2.0.18 windows')
def test_get_grab(self):
    """Ensure get_grab() works as expected"""
    surf = pygame.display.set_mode((10, 10))
    for i in range(5):
        pygame.event.set_grab(i % 2)
        self.assertEqual(pygame.event.get_grab(), i % 2)