import os
from queuelib.pqueue import PriorityQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
def test_reopen_with_prio(self):
    q1 = PriorityQueue(self.qfactory)
    q1.push(b'a', 3)
    q1.push(b'b', 1)
    q1.push(b'c', 2)
    active = q1.close()
    q2 = PriorityQueue(self.qfactory, startprios=active)
    self.assertEqual(q2.pop(), b'b')
    self.assertEqual(q2.pop(), b'c')
    self.assertEqual(q2.pop(), b'a')
    self.assertEqual(q2.close(), [])