from lazyops.utils import resolve_missing, require_missing_wrapper
def resolve_croniter(required: bool=False):
    """
    Ensures that `croniter` is available
    """
    global croniter, _croniter_available
    if not _croniter_available:
        resolve_missing('croniter', required=required)
        from croniter import croniter
        _croniter_available = True