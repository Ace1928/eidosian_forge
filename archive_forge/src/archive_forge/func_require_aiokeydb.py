from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def require_aiokeydb(required: bool=False, version: str=_min_version):
    """
    Wrapper for `resolve_aiokeydb` that can be used as a decorator
    """

    def decorator(func):
        return require_missing_wrapper(resolver=resolve_aiokeydb, func=func, required=required, version=version)
    return decorator