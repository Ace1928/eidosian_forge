import collections
import time
import unittest
import os
import pygame
def test_event_attribute(self):
    e1 = pygame.event.Event(pygame.USEREVENT, attr1='attr1')
    self.assertEqual(e1.attr1, 'attr1')