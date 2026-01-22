import collections
import time
import unittest
import os
import pygame
def test_event_numevents(self):
    """Ensures NUMEVENTS does not exceed the maximum SDL number of events."""
    MAX_SDL_EVENTS = 65535
    self.assertLessEqual(pygame.NUMEVENTS, MAX_SDL_EVENTS)