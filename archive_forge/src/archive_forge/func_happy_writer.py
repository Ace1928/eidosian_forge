import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def happy_writer():
    with lock.write_lock():
        activated.append(lock.owner)