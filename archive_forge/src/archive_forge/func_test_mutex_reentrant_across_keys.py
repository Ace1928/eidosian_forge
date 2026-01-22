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
def test_mutex_reentrant_across_keys(self):
    backend = self._backend()
    for x in range(3):
        m1 = backend.get_mutex('foo')
        m2 = backend.get_mutex('bar')
        try:
            m1.acquire()
            assert m2.acquire(False)
            assert not m2.acquire(False)
            m2.release()
            assert m2.acquire(False)
            assert not m2.acquire(False)
            m2.release()
        finally:
            m1.release()