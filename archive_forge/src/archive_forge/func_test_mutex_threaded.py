import collections
import itertools
import json
import random
from threading import Lock
from threading import Thread
import time
import uuid
import pytest
from dogpile.cache import CacheRegion
from dogpile.cache import register_backend
from dogpile.cache.api import CacheBackend
from dogpile.cache.api import CacheMutex
from dogpile.cache.api import CantDeserializeException
from dogpile.cache.api import NO_VALUE
from dogpile.cache.region import _backend_loader
from .assertions import assert_raises_message
from .assertions import eq_
@pytest.mark.time_intensive
def test_mutex_threaded(self):
    backend = self._backend()
    backend.get_mutex('foo')
    lock = Lock()
    canary = []

    def f():
        for x in range(5):
            mutex = backend.get_mutex('foo')
            mutex.acquire()
            for y in range(5):
                ack = lock.acquire(False)
                canary.append(ack)
                time.sleep(0.002)
                if ack:
                    lock.release()
            mutex.release()
            time.sleep(0.02)
    threads = [Thread(target=f) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert False not in canary