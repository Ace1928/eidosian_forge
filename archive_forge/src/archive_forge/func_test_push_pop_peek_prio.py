import os
from queuelib.pqueue import PriorityQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
def test_push_pop_peek_prio(self):
    self.assertEqual(self.q.peek(), None)
    self.q.push(b'a', 3)
    self.q.push(b'b', 1)
    self.q.push(b'c', 2)
    self.q.push(b'd', 1)
    self.assertEqual(self.q.peek(), b'd')
    self.assertEqual(self.q.pop(), b'd')
    self.assertEqual(self.q.peek(), b'b')
    self.assertEqual(self.q.pop(), b'b')
    self.assertEqual(self.q.peek(), b'c')
    self.assertEqual(self.q.pop(), b'c')
    self.assertEqual(self.q.peek(), b'a')
    self.assertEqual(self.q.pop(), b'a')
    self.assertEqual(self.q.peek(), None)
    self.assertEqual(self.q.pop(), None)