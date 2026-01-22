import collections
import time
import unittest
import os
import pygame
def test_peek__empty_queue(self):
    """Ensure peek() works correctly on an empty queue."""
    pygame.event.clear()
    peeked = pygame.event.peek()
    self.assertFalse(peeked)
    for event_type in EVENT_TYPES:
        peeked = pygame.event.peek(event_type)
        self.assertFalse(peeked)
    peeked = pygame.event.peek(EVENT_TYPES)
    self.assertFalse(peeked)