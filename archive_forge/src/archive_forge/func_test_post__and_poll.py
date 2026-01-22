import collections
import time
import unittest
import os
import pygame
def test_post__and_poll(self):
    """Ensure events can be posted to the queue."""
    e1 = pygame.event.Event(pygame.USEREVENT, attr1='attr1')
    pygame.event.post(e1)
    posted_event = pygame.event.poll()
    self.assertEqual(e1.attr1, posted_event.attr1, race_condition_notification)
    for i in range(1, 13):
        pygame.event.post(pygame.event.Event(EVENT_TYPES[i], **EVENT_TEST_PARAMS[EVENT_TYPES[i]]))
        self.assertEqual(pygame.event.poll().type, EVENT_TYPES[i], race_condition_notification)