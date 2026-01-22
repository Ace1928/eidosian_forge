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
def test_region_expire(self):
    some_key = gen_some_key()
    expire_time = 1.0
    reg = self._region(config_args={'expiration_time': expire_time})
    counter = itertools.count(1)

    def creator():
        return 'some value %d' % next(counter)
    eq_(reg.get_or_create(some_key, creator), 'some value 1')
    time.sleep(expire_time + 0.2 * expire_time)
    post_expiration = reg.get(some_key, ignore_expiration=True)
    if post_expiration is not NO_VALUE:
        eq_(post_expiration, 'some value 1')
    eq_(reg.get_or_create(some_key, creator), 'some value 2')
    eq_(reg.get(some_key), 'some value 2')