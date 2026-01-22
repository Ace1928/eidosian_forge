import collections
import time
import unittest
import os
import pygame
def test_get_blocked__event_sequence(self):
    """Ensure get_blocked() can handle a sequence of event types."""
    event_types = [pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.WINDOWMINIMIZED, pygame.USEREVENT]
    blocked = pygame.event.get_blocked(event_types)
    self.assertFalse(blocked)
    pygame.event.set_blocked(event_types[2])
    blocked = pygame.event.get_blocked(event_types)
    self.assertTrue(blocked)
    pygame.event.set_blocked(event_types)
    blocked = pygame.event.get_blocked(event_types)
    self.assertTrue(blocked)