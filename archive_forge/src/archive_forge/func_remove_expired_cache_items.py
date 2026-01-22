import datetime
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk.consts import TYPE_CHECKING
def remove_expired_cache_items():
    """
    Remove expired cache items from the cache.
    """
    global EXPLAIN_CACHE
    now = datetime_utcnow()
    for key, expiration_time in EXPLAIN_CACHE.items():
        expiration_in_the_past = expiration_time < now
        if expiration_in_the_past:
            del EXPLAIN_CACHE[key]