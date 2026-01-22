import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._multiprocessing_helpers import mp
@staticmethod
def in_main_thread():
    return isinstance(threading.current_thread(), threading._MainThread)