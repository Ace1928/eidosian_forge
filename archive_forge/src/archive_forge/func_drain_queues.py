from collections import deque
from threading import local
def drain_queues(self):
    assert self.is_tick_used
    self.drain_queue(self.normal_queue)
    self.reset()
    self.have_drained_queues = True
    self.drain_queue(self.late_queue)