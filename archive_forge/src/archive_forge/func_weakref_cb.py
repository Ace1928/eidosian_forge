from concurrent.futures import _base
import itertools
import queue
import threading
import types
import weakref
import os
def weakref_cb(_, q=self._work_queue):
    q.put(None)