import collections
import time
import unittest
import os
import pygame
def test_event_name__kwargs(self):
    """Ensure event_name() returns the correct event name when kwargs used."""
    for expected_name, event in NAMES_AND_EVENTS:
        self.assertEqual(pygame.event.event_name(type=event), expected_name, f'0x{event:X}')