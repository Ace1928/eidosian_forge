from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_niquests(required: bool=False):
    """
    Ensures that `niquests` is available
    """
    global niquests, _niquests_available
    if not _niquests_available:
        resolve_missing('niquests', required=required)
        import niquests
        _niquests_available = True
        globals()['niquests'] = niquests