import unittest
from os.path import abspath, dirname, join
import errno
import os
def test_redis_storage(self):
    if os.environ.get('NONETWORK'):
        return
    try:
        from kivy.storage.redisstore import RedisStore
        from redis.exceptions import ConnectionError
        try:
            params = dict(db=15)
            self._do_store_test_empty(RedisStore(params))
            self._do_store_test_filled(RedisStore(params))
        except ConnectionError:
            pass
    except ImportError:
        pass