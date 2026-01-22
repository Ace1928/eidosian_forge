import os
from queuelib.pqueue import PriorityQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
def test_popped_internal_queues_closed(self):
    self.q.push(b'a', 3)
    self.q.push(b'b', 1)
    self.q.push(b'c', 2)
    p1queue = self.q.queues[1]
    self.assertEqual(self.q.pop(), b'b')
    self.q.close()
    assert p1queue.closed