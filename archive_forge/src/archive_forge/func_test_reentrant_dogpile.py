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
def test_reentrant_dogpile(self):
    reg = self._region()

    def create_foo():
        return 'foo' + reg.get_or_create('bar', create_bar)

    def create_bar():
        return 'bar'
    eq_(reg.get_or_create('foo', create_foo), 'foobar')
    eq_(reg.get_or_create('foo', create_foo), 'foobar')