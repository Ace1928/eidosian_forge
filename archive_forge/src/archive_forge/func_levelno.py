import functools
import json
import multiprocessing
import os
import threading
from contextlib import contextmanager
from threading import Thread
from ._colorizer import Colorizer
from ._locks_machinery import create_handler_lock
@property
def levelno(self):
    return self._levelno