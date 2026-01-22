import os
from concurrent.futures import _base
import queue
import multiprocessing as mp
import multiprocessing.connection
from multiprocessing.queues import Queue
import threading
import weakref
from functools import partial
import itertools
import sys
from traceback import format_exception
def shutdown_workers(self):
    n_children_to_stop = self.get_n_children_alive()
    n_sentinels_sent = 0
    while n_sentinels_sent < n_children_to_stop and self.get_n_children_alive() > 0:
        for i in range(n_children_to_stop - n_sentinels_sent):
            try:
                self.call_queue.put_nowait(None)
                n_sentinels_sent += 1
            except queue.Full:
                break