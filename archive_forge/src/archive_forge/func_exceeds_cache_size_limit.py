import logging
import types
import weakref
from dataclasses import dataclass
from . import config
def exceeds_cache_size_limit(cache_size: CacheSizeRelevantForFrame) -> bool:
    """
    Checks if we are exceeding the cache size limit.
    """
    return cache_size.will_compilation_exceed(config.cache_size_limit)