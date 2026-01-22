from __future__ import annotations
import abc
import hashlib
from fsspec.implementations.local import make_path_posix
def create_cache_mapper(same_names: bool) -> AbstractCacheMapper:
    """Factory method to create cache mapper for backward compatibility with
    ``CachingFileSystem`` constructor using ``same_names`` kwarg.
    """
    if same_names:
        return BasenameCacheMapper()
    else:
        return HashCacheMapper()