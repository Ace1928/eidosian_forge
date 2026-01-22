import collections
import time
import unittest
import os
import pygame
def test_get__event_sequence(self):
    """Ensure get() can handle a sequence of event types."""
    event_types = [pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEMOTION]
    other_event_type = pygame.MOUSEBUTTONUP
    expected_events = []
    pygame.event.clear()
    retrieved_events = pygame.event.get(event_types)
    self._assertExpectedEvents(expected=expected_events, got=retrieved_events)
    expected_events = []
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(other_event_type, **EVENT_TEST_PARAMS[other_event_type]))
    retrieved_events = pygame.event.get(event_types)
    self._assertExpectedEvents(expected=expected_events, got=retrieved_events)
    expected_events = [pygame.event.Event(event_types[0], **EVENT_TEST_PARAMS[event_types[0]])]
    pygame.event.clear()
    pygame.event.post(expected_events[0])
    retrieved_events = pygame.event.get(event_types)
    self._assertExpectedEvents(expected=expected_events, got=retrieved_events)
    pygame.event.clear()
    expected_events = []
    for etype in event_types:
        expected_events.append(pygame.event.Event(etype, **EVENT_TEST_PARAMS[etype]))
        pygame.event.post(expected_events[-1])
    retrieved_events = pygame.event.get(event_types)
    self._assertExpectedEvents(expected=expected_events, got=retrieved_events)