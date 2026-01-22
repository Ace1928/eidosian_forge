import os
import platform
import unittest
import pygame
import time
@unittest.skipIf(platform.machine() == 's390x', 'Fails on s390x')
@unittest.skipIf(os.environ.get('CI', None), 'CI can have variable time slices, slow.')
def test_set_timer(self):
    """Tests time.set_timer()"""
    '\n        Tests if a timer will post the correct amount of eventid events in\n        the specified delay. Test is posting event objects work.\n        Also tests if setting milliseconds to 0 stops the timer and if\n        the once argument and repeat arguments work.\n        '
    pygame.init()
    TIMER_EVENT_TYPE = pygame.event.custom_type()
    timer_event = pygame.event.Event(TIMER_EVENT_TYPE)
    delta = 50
    timer_delay = 100
    test_number = 8
    events = 0
    pygame.event.clear()
    pygame.time.set_timer(TIMER_EVENT_TYPE, timer_delay)
    t1 = pygame.time.get_ticks()
    max_test_time = t1 + timer_delay * test_number + delta
    while events < test_number:
        for event in pygame.event.get():
            if event == timer_event:
                events += 1
        if pygame.time.get_ticks() > max_test_time:
            break
    pygame.time.set_timer(TIMER_EVENT_TYPE, 0)
    t2 = pygame.time.get_ticks()
    self.assertEqual(events, test_number)
    self.assertAlmostEqual(timer_delay * test_number, t2 - t1, delta=delta)
    pygame.time.delay(200)
    self.assertNotIn(timer_event, pygame.event.get())
    pygame.time.set_timer(TIMER_EVENT_TYPE, timer_delay)
    pygame.time.delay(int(timer_delay * 3.5))
    self.assertEqual(pygame.event.get().count(timer_event), 3)
    pygame.time.set_timer(TIMER_EVENT_TYPE, timer_delay * 10)
    pygame.time.delay(timer_delay * 5)
    self.assertNotIn(timer_event, pygame.event.get())
    pygame.time.set_timer(TIMER_EVENT_TYPE, timer_delay * 3)
    pygame.time.delay(timer_delay * 7)
    self.assertEqual(pygame.event.get().count(timer_event), 2)
    pygame.time.set_timer(TIMER_EVENT_TYPE, timer_delay)
    pygame.time.delay(int(timer_delay * 5.5))
    self.assertEqual(pygame.event.get().count(timer_event), 5)
    pygame.time.set_timer(TIMER_EVENT_TYPE, 10, True)
    pygame.time.delay(40)
    self.assertEqual(pygame.event.get().count(timer_event), 1)
    events_to_test = [pygame.event.Event(TIMER_EVENT_TYPE), pygame.event.Event(TIMER_EVENT_TYPE, foo='9gwz5', baz=12, lol=[124, (34, '')]), pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, unicode='a')]
    repeat = 3
    millis = 50
    for e in events_to_test:
        pygame.time.set_timer(e, millis, loops=repeat)
        pygame.time.delay(2 * millis * repeat)
        self.assertEqual(pygame.event.get().count(e), repeat)
    pygame.quit()