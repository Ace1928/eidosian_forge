import os
import sys
import signal
import itertools
import threading
from _weakrefset import WeakSet
def parent_process():
    """
    Return process object representing the parent process
    """
    return _parent_process