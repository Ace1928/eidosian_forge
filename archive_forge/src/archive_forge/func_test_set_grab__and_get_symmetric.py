import collections
import time
import unittest
import os
import pygame
@unittest.skip('flaky test, and broken on 2.0.18 windows')
def test_set_grab__and_get_symmetric(self):
    """Ensure event grabbing can be enabled and disabled.

        WARNING: Moving the mouse off the display during this test can cause it
                 to fail.
        """
    surf = pygame.display.set_mode((10, 10))
    pygame.event.set_grab(True)
    self.assertTrue(pygame.event.get_grab())
    pygame.event.set_grab(False)
    self.assertFalse(pygame.event.get_grab())