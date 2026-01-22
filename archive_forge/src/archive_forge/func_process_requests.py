import collections
import errno
import heapq
import logging
import math
import os
import pyngus
import select
import socket
import threading
import time
import uuid
def process_requests(self):
    """Invoked by the eventloop thread, execute each queued callable."""
    with self._pipe_lock:
        if not self._pipe_ready:
            return
        self._pipe_ready = False
        os.read(self._wakeup_pipe[0], 512)
        requests = self._requests
        self._requests = collections.deque()
    for r in requests:
        r()