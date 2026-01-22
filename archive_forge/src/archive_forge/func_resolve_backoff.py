from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_backoff(required: bool=False):
    """
    Ensures that `backoff` is available
    """
    global backoff, _backoff_available
    if not _backoff_available:
        resolve_missing('backoff', required=required)
        import backoff
        _backoff_available = True
        globals()['backoff'] = backoff