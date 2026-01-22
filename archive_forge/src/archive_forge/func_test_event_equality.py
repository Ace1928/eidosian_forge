import collections
import time
import unittest
import os
import pygame
def test_event_equality(self):
    """Ensure that events can be compared correctly."""
    a = pygame.event.Event(EVENT_TYPES[0], a=1)
    b = pygame.event.Event(EVENT_TYPES[0], a=1)
    c = pygame.event.Event(EVENT_TYPES[1], a=1)
    d = pygame.event.Event(EVENT_TYPES[0], a=2)
    self.assertTrue(a == a)
    self.assertFalse(a != a)
    self.assertTrue(a == b)
    self.assertFalse(a != b)
    self.assertTrue(a != c)
    self.assertFalse(a == c)
    self.assertTrue(a != d)
    self.assertFalse(a == d)