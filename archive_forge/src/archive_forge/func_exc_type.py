import contextlib
import inspect
import multiprocessing
import sys
import threading
from time import monotonic
import traceback
@property
def exc_type(self):
    return self.exc_info[0]