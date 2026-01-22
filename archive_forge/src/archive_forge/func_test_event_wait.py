import collections
import time
import unittest
import os
import pygame
def test_event_wait(self):
    """Ensure wait() waits for an event on the queue."""
    event = pygame.event.Event(EVENT_TYPES[0], **EVENT_TEST_PARAMS[EVENT_TYPES[0]])
    pygame.event.post(event)
    wait_event = pygame.event.wait()
    self.assertEqual(wait_event.type, event.type)
    wait_event = pygame.event.wait(100)
    self.assertEqual(wait_event.type, pygame.NOEVENT)
    event = pygame.event.Event(EVENT_TYPES[0], **EVENT_TEST_PARAMS[EVENT_TYPES[0]])
    pygame.event.post(event)
    wait_event = pygame.event.wait(100)
    self.assertEqual(wait_event.type, event.type)
    pygame.time.set_timer(pygame.USEREVENT, 50, 3)
    for wait_time, expected_type, expected_time in ((60, pygame.USEREVENT, 50), (65, pygame.USEREVENT, 50), (20, pygame.NOEVENT, 20), (45, pygame.USEREVENT, 30), (70, pygame.NOEVENT, 70)):
        start_time = time.perf_counter()
        self.assertEqual(pygame.event.wait(wait_time).type, expected_type)
        self.assertAlmostEqual(time.perf_counter() - start_time, expected_time / 1000, delta=0.01)
    pygame.time.set_timer(pygame.USEREVENT, 100, 1)
    start_time = time.perf_counter()
    self.assertEqual(pygame.event.wait().type, pygame.USEREVENT)
    self.assertAlmostEqual(time.perf_counter() - start_time, 0.1, delta=0.01)
    pygame.time.set_timer(pygame.USEREVENT, 50, 1)
    self.assertEqual(pygame.event.wait(40).type, pygame.NOEVENT)