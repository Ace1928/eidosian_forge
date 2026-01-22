import collections
import time
import unittest
import os
import pygame
def test_set_allowed(self):
    """Ensure a blocked event type can be unblocked/allowed."""
    event = EVENT_TYPES[0]
    pygame.event.set_blocked(event)
    self.assertTrue(pygame.event.get_blocked(event))
    pygame.event.set_allowed(event)
    self.assertFalse(pygame.event.get_blocked(event))