from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_bs4(required: bool=False):
    """
    Ensures that `bs4` is available
    """
    global bs4, _bs4_available
    if not _bs4_available:
        resolve_missing('bs4', required=required)
        import bs4
        _bs4_available = True
        globals()['bs4'] = bs4