import collections
import time
import unittest
import os
import pygame
def test_event_bool(self):
    self.assertFalse(pygame.event.Event(pygame.NOEVENT))
    for event_type in [pygame.MOUSEBUTTONDOWN, pygame.ACTIVEEVENT, pygame.WINDOWLEAVE, pygame.USEREVENT_DROPFILE]:
        self.assertTrue(pygame.event.Event(event_type))