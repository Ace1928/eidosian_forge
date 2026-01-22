import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def double_bad_reader():
    with lock.read_lock():
        with lock.read_lock():
            raise RuntimeError('Broken')