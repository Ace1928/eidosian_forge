import collections
import itertools as it
import re
import threading
from repoze.lru import LRUCache
import six
from routes import request_config
from routes.util import (
from routes.route import Route
def create_regs(self, *args, **kwargs):
    """Atomically creates regular expressions for all connected
        routes
        """
    self.create_regs_lock.acquire()
    try:
        self._create_regs(*args, **kwargs)
    finally:
        self.create_regs_lock.release()