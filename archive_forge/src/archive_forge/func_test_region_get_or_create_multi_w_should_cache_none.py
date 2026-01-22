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
def test_region_get_or_create_multi_w_should_cache_none(self):
    reg = self._region()
    values = reg.get_or_create_multi(['key1', 'key2', 'key3'], lambda *k: [None, None, None], should_cache_fn=lambda v: v is not None)
    eq_(values, [None, None, None])