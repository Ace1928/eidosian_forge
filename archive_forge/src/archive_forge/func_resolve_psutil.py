from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_psutil(required: bool=False):
    """
    Ensures that `psutil` is available
    """
    global psutil, _psutil_available
    if not _psutil_available:
        resolve_missing('psutil', required=required)
        import psutil
        _psutil_available = True
        globals()['psutil'] = psutil