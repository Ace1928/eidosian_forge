from mako.cache import CacheImpl
def get_and_replace(self, key, creation_function, **kw):
    expiration_time = kw.pop('timeout', None)
    return self._get_region(**kw).get_or_create(key, creation_function, expiration_time=expiration_time)