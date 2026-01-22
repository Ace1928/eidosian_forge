import collections
import time
import unittest
import os
import pygame
def test_get_type(self):
    ev = pygame.event.Event(pygame.USEREVENT)
    pygame.event.post(ev)
    queue = pygame.event.get(pygame.USEREVENT)
    self.assertEqual(len(queue), 1)
    self.assertEqual(queue[0].type, pygame.USEREVENT)
    TESTEVENTS = 10
    for _ in range(TESTEVENTS):
        pygame.event.post(ev)
    q = pygame.event.get([pygame.USEREVENT])
    self.assertEqual(len(q), TESTEVENTS)
    for event in q:
        self.assertEqual(event, ev)