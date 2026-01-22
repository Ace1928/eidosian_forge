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
def join_executor_internals(self):
    self.shutdown_workers()
    self.call_queue.close()
    self.call_queue.join_thread()
    with self.shutdown_lock:
        self.thread_wakeup.close()
    for p in self.processes.values():
        p.join()