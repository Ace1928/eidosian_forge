import atexit
from concurrent.futures import _base
import Queue as queue
import multiprocessing
import threading
import weakref
import sys
def shutdown_one_process():
    """Tell a worker to terminate, which will in turn wake us again"""
    call_queue.put(None)
    nb_shutdown_processes[0] += 1