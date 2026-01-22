import collections
import time
import unittest
import os
import pygame
def test_clear__empty_queue(self):
    """Ensure clear() works correctly on an empty queue."""
    expected_events = []
    pygame.event.clear()
    pygame.event.clear()
    retrieved_events = pygame.event.get()
    self.assertListEqual(retrieved_events, expected_events)