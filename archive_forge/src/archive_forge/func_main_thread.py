import os as _os
import sys as _sys
import _thread
import functools
from time import monotonic as _time
from _weakrefset import WeakSet
from itertools import islice as _islice, count as _count
from _thread import stack_size
def main_thread():
    """Return the main thread object.

    In normal conditions, the main thread is the thread from which the
    Python interpreter was started.
    """
    return _main_thread