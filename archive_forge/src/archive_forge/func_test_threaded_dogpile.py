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
def test_threaded_dogpile(self):
    reg = self._region(config_args={'expiration_time': 0.25})
    lock = Lock()
    canary = []
    some_key = gen_some_key()

    def creator():
        ack = lock.acquire(False)
        canary.append(ack)
        time.sleep(0.25)
        if ack:
            lock.release()
        return 'some value'

    def f():
        for x in range(5):
            reg.get_or_create(some_key, creator)
            time.sleep(0.5)
    threads = [Thread(target=f) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(canary) > 2
    if not reg.backend.has_lock_timeout():
        assert False not in canary