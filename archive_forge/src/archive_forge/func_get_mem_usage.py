import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def get_mem_usage():
    """
    Returns the internal memory usage of the profiler itself.
    """
    return _yappi.get_mem_usage()