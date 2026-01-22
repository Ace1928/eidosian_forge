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
def test_decorated_fn_functionality(self):
    reg = self._region()
    counter = itertools.count(1)

    @reg.cache_on_arguments()
    def my_function(x, y):
        return next(counter) + x + y
    my_function.invalidate(3, 4)
    my_function.invalidate(5, 6)
    my_function.invalidate(4, 3)
    eq_(my_function(3, 4), 8)
    eq_(my_function(5, 6), 13)
    eq_(my_function(3, 4), 8)
    eq_(my_function(4, 3), 10)
    my_function.invalidate(4, 3)
    eq_(my_function(4, 3), 11)