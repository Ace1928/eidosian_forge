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
def get_clock_type():
    """
    Returns the underlying clock type
    """
    return _yappi.get_clock_type()