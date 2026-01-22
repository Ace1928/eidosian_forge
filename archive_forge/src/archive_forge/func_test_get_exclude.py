import collections
import time
import unittest
import os
import pygame
def test_get_exclude(self):
    pygame.event.post(pygame.event.Event(pygame.USEREVENT))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN))
    queue = pygame.event.get(exclude=pygame.KEYDOWN)
    self.assertEqual(len(queue), 1)
    self.assertEqual(queue[0].type, pygame.USEREVENT)
    pygame.event.post(pygame.event.Event(pygame.KEYUP))
    pygame.event.post(pygame.event.Event(pygame.USEREVENT))
    queue = pygame.event.get(exclude=(pygame.KEYDOWN, pygame.KEYUP))
    self.assertEqual(len(queue), 1)
    self.assertEqual(queue[0].type, pygame.USEREVENT)
    queue = pygame.event.get()
    self.assertEqual(len(queue), 2)