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
def test_mutex(self):
    backend = self._backend()
    mutex = backend.get_mutex('foo')
    assert not mutex.locked()
    ac = mutex.acquire()
    assert ac
    ac2 = mutex.acquire(False)
    assert mutex.locked()
    assert not ac2
    mutex.release()
    assert not mutex.locked()
    ac3 = mutex.acquire()
    assert ac3
    mutex.release()