import datetime
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk.consts import TYPE_CHECKING
def should_run_explain_plan(statement, options):
    """
    Check cache if the explain plan for the given statement should be run.
    """
    global EXPLAIN_CACHE
    remove_expired_cache_items()
    key = hash(statement)
    if key in EXPLAIN_CACHE:
        return False
    explain_cache_size = options.get('explain_cache_size', EXPLAIN_CACHE_SIZE)
    cache_is_full = len(EXPLAIN_CACHE.keys()) >= explain_cache_size
    if cache_is_full:
        return False
    return True