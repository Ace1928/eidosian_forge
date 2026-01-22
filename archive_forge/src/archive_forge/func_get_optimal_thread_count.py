import _thread
import collections
import multiprocessing
import threading
from taskflow.utils import misc
def get_optimal_thread_count(default=2):
    """Try to guess optimal thread count for current system."""
    try:
        return multiprocessing.cpu_count() + 1
    except NotImplementedError:
        return default