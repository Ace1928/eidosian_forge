import asyncore
from collections import deque
from warnings import _deprecated
def discard_buffers(self):
    self.ac_in_buffer = b''
    del self.incoming[:]
    self.producer_fifo.clear()