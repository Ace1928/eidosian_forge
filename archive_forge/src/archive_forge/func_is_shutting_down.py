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
def is_shutting_down(self):
    executor = self.executor_reference()
    return _global_shutdown or executor is None or executor._shutdown_thread