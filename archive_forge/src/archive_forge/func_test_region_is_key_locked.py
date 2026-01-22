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
def test_region_is_key_locked(self):
    reg = self._region()
    random_key = str(uuid.uuid1())
    assert not reg.get(random_key)
    eq_(reg.key_is_locked(random_key), False)
    eq_(reg.key_is_locked(random_key), False)
    mutex = reg.backend.get_mutex(random_key)
    if mutex:
        mutex.acquire()
        eq_(reg.key_is_locked(random_key), True)
        mutex.release()
        eq_(reg.key_is_locked(random_key), False)