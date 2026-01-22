import os
import pickle
import re
import requests
import sys
import time
from datetime import datetime
from functools import wraps
from tempfile import gettempdir
def cache_results(function):
    """Return decorated function that caches the results."""

    def save_to_permacache():
        """Save the in-memory cache data to the permacache.

        There is a race condition here between two processes updating at the
        same time. It's perfectly acceptable to lose and/or corrupt the
        permacache information as each process's in-memory cache will remain
        in-tact.

        """
        update_from_permacache()
        try:
            with open(filename, 'wb') as fp:
                pickle.dump(cache, fp, pickle.HIGHEST_PROTOCOL)
        except IOError:
            pass

    def update_from_permacache():
        """Attempt to update newer items from the permacache."""
        try:
            with open(filename, 'rb') as fp:
                permacache = pickle.load(fp)
        except Exception:
            return
        for key, value in permacache.items():
            if key not in cache or value[0] > cache[key][0]:
                cache[key] = value
    cache = {}
    cache_expire_time = 3600
    try:
        filename = os.path.join(gettempdir(), 'update_checker_cache.pkl')
        update_from_permacache()
    except NotImplementedError:
        filename = None

    @wraps(function)
    def wrapped(obj, package_name, package_version, **extra_data):
        """Return cached results if available."""
        now = time.time()
        key = (package_name, package_version)
        if not obj._bypass_cache and key in cache:
            cache_time, retval = cache[key]
            if now - cache_time < cache_expire_time:
                return retval
        retval = function(obj, package_name, package_version, **extra_data)
        cache[key] = (now, retval)
        if filename:
            save_to_permacache()
        return retval
    return wrapped