import collections
import time
import unittest
import os
import pygame
def test_Event(self):
    """Ensure an Event object can be created."""
    e = pygame.event.Event(pygame.USEREVENT, some_attr=1, other_attr='1')
    self.assertEqual(e.some_attr, 1)
    self.assertEqual(e.other_attr, '1')
    self.assertEqual(e.type, pygame.USEREVENT)
    self.assertIs(e.dict, e.__dict__)
    e.some_attr = 12
    self.assertEqual(e.some_attr, 12)
    e.new_attr = 15
    self.assertEqual(e.new_attr, 15)
    self.assertRaises(AttributeError, setattr, e, 'type', 0)
    self.assertRaises(AttributeError, setattr, e, 'dict', None)
    d = dir(e)
    attrs = ('type', 'dict', '__dict__', 'some_attr', 'other_attr', 'new_attr')
    for attr in attrs:
        self.assertIn(attr, d)
    self.assertRaises(ValueError, pygame.event.Event, 10, type=100)