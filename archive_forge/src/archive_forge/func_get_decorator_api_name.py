from __future__ import annotations
import enum
def get_decorator_api_name(cache_type: CacheType) -> str:
    """Return the name of the public decorator API for the given CacheType."""
    if cache_type is CacheType.DATA:
        return 'cache_data'
    if cache_type is CacheType.RESOURCE:
        return 'cache_resource'
    raise RuntimeError(f"Unrecognized CacheType '{cache_type}'")