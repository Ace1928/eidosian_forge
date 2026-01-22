import os
import platform
import unittest
import pygame
import time
@unittest.skipIf(platform.machine() == 's390x', 'Fails on s390x')
@unittest.skipIf(os.environ.get('CI', None), 'CI can have variable time slices, slow.')
def test_get_time(self):
    delay = 0.1
    delay_miliseconds = delay * 10 ** 3
    iterations = 10
    delta = 50
    c = Clock()
    self.assertEqual(c.get_time(), 0)
    for i in range(iterations):
        time.sleep(delay)
        c.tick()
        c1 = c.get_time()
        self.assertAlmostEqual(delay_miliseconds, c1, delta=delta)
    for i in range(iterations):
        t0 = time.time()
        time.sleep(delay)
        c.tick()
        t1 = time.time()
        c1 = c.get_time()
        d0 = (t1 - t0) * 10 ** 3
        self.assertAlmostEqual(d0, c1, delta=delta)