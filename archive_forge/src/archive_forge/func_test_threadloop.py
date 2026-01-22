import unittest
from pygame.threads import FuncResult, tmap, WorkerQueue, Empty, STOP
from pygame import threads, Surface, transform
import time
def test_threadloop(self):
    wq = WorkerQueue(1)
    wq.do(wq.threadloop)
    l = []
    wq.do(l.append, 1)
    time.sleep(0.5)
    self.assertEqual(l[0], 1)
    wq.stop()
    self.assertFalse(wq.pool[0].is_alive())